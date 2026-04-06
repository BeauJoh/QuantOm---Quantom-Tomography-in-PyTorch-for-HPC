import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_proxy_theory_lite import TorchProxyTheoryLite
from torch_2D_inverse_transform_sampler import Torch2DInverseTransformSampler
from scipy.stats import describe
import time
import os
import argparse
import ctypes
from pathlib import Path

#preload
if os.environ.get('HOST') == "faraday":
    ctypes.CDLL("/opt/rocm-7.0.1/lib/libamdhip64.so", mode=ctypes.RTLD_GLOBAL)

'''
This script runs 2D LOITS on the proxy app lite theory, i.e. 5 parameters are required to produce a 2D distribution.
The following settings (see argparse below) are important:

--torch_device: Sets the device you run  on: cpu, cuda, mps,....
--threading: Flag to use multi-threading within LOTS. Default is False. 
--grid_size: Gird size in x and y dimension. The larger the smoother are the sampled distributions. 

Impact of threading: 
(a) Threading OFF: LOITS samples: n_gen_events * batch_size events e.g.: n_gen_events = 2000 and batch_size = 5 ==> 10000 events are sampled
(b) Threading ON: LOITS will run batch_size threads, each producing: n_gen_events / batch_size samples e.g. batch_size = 20 and n_gen_events = 100 ==> LOITS uses 20 threads and each thread samples 5 events

Output of this script:
.npy arrays that show the time as a function of batch_size. It is up to the user to turn these into pretty plots

For questions and concerns, please contact:
Daniel Lersch - dlersch@jlab.org
'''
torch.set_default_dtype(torch.float64)

#enable the following to perform a comparison against the SYCL implementation of 2D-LOITS
RUN_PYTORCH=True
RUN_CPP=True
RUN_OMP=True
RUN_SYCL=True
hostname = os.getenv('HOST')

COLLECT_MEMORY_FOOTPRINT=False

parser = argparse.ArgumentParser(prog="scan_loits_performance.py")

parser.add_argument("--torch_device",type=str,default="cpu",help="Torch device to run on")
parser.add_argument("--n_trials",type=int,default=5,help="Number of trials per scan")
parser.add_argument("--logdir",type=str,default="results_loits_performance_scans",help="Directory to store results")
parser.add_argument("--n_gen_events",type=int,default=2000,help="Number of events to generate")
parser.add_argument("--grid_size",type=int,default=10,help="Grid size for sampling")
parser.add_argument("--threading",type=bool,default=False,help="Use sampler with threading")
parser.add_argument("--pytorch_sampler",default=False,help="Store the timing results of the PyTorch based sampler?",action='store_true')
parser.add_argument("--cpp_sampler",default=False,help="Use the hand rolled CPP-based sampler?",action='store_true')
parser.add_argument("--omp_sampler",default=False,help="Use the hand rolled OpenMP-based sampler?",action='store_true')
parser.add_argument("--sycl_sampler",default=False,help="Use the hand rolled SYCL-based sampler?",action='store_true')
parser.add_argument("--sycl_implementations",default=False,help="CSV list of SYCL implementations to evaluate",type=str)
parser.add_argument("--disable_gradient_tracking",default=False,help="enable backward propagation---Required to do useful work",action='store_true')
parser.add_argument("--collect_memory_footprint",default=False,help="show max device memory used for the given problem size",action='store_true')
parser.add_argument("--internal_timing",type=str,default="",help="time and present the internals of the PyTorch sampler?")

args = parser.parse_args()

print("Retreive user input...")

grid_size = args.grid_size
n_gen_events = args.n_gen_events
torch_device = args.torch_device
n_trials = args.n_trials
logdir = args.logdir
threading = args.threading
RUN_PYTORCH = args.pytorch_sampler
RUN_CPP = args.cpp_sampler
RUN_OMP = args.omp_sampler
RUN_SYCL = args.sycl_sampler
SYCL_IMPLS = ['acpp']
if args.sycl_implementations:
    SYCL_IMPLS = args.sycl_implementations.split(',')
requires_grad = not args.disable_gradient_tracking
internal_timing = args.internal_timing
COLLECT_MEMORY_FOOTPRINT=args.collect_memory_footprint

print("...done!")
print(" ")

print("Create result directory...")

os.makedirs(logdir,exist_ok=True)

print("...done!")
print(" ")

#if we haven't collected any memory usage yet
if not Path("memory_footprints.csv").is_file():
    with open('memory_footprints.csv', 'w') as the_file:
        the_file.write('n_gen_events,memory\n')

# Load the theory and sampler:
print("Load theory and sampler module...")

t_cfg = {
          'n_points_x':grid_size,
          'n_points_y':grid_size,
          'n_cdf_points_x': 10,
          'n_cdf_points_y': 10,
          'average': False,
        }

theory = TorchProxyTheoryLite(config=t_cfg,devices=torch_device)

s_cfg = {
        'use_threading': threading,
        'requires_grad': requires_grad,
}

#run the reference pytorch implementation if no others sampler implementations are set
if RUN_PYTORCH or (not RUN_CPP and not RUN_OMP and not RUN_SYCL):
    sampler = Torch2DInverseTransformSampler(config=s_cfg,devices=torch_device,internal_logging=internal_timing)

if RUN_CPP:
    from cpp import cpp_wrapper
    cpp_sampler = cpp_wrapper.cpp_sampler(internal_timing)

if RUN_OMP:
    from omp import omp_wrapper
    omp_sampler = omp_wrapper.omp_sampler(internal_timing)

if RUN_SYCL:
    sycl_samplers = []
    for SYCL_IMPL in SYCL_IMPLS:
        if SYCL_IMPL == 'acpp':
            from sycl.acpp import sycl_wrapper as sycl_wrapper
        if SYCL_IMPL == 'dpcp':
            from sycl.dpcp import sycl_wrapper as sycl_wrapper
        sycl_sampler = sycl_wrapper.sycl_sampler(internal_timing)
        sycl_samplers.append({'sampler':sycl_sampler,'name':SYCL_IMPL})

print("...done!")
print(" ")

print("Prepare for scans...")

tparam = np.array([0.67,0.2,0.23,0.67,0.5])    
true_params = torch.as_tensor(tparam,device=torch_device,dtype=torch.float32)
true_params = torch.unsqueeze(true_params,0)

print("...done!")
print(" ")

print("Run scans...")

#batch_sizes = [1,2,4,6,8,10,15,20,25,30,35,40,45,50]
batch_sizes = [1]
n_scans = len(batch_sizes)

print(" ")
py_avg_times, cpp_avg_times, omp_avg_times, sycl_avg_times = [],[],[],[]

problem_size_vs_memory = []
import pandas as pd
df = pd.DataFrame()
#+++++++++++++++++++++++++++
#run JIT for AdaptiveCPP's generic compiler to avoid timing compilation overhead

if RUN_SYCL: 
    for sycl_sampler in sycl_samplers:
        sycl_sampler = sycl_sampler["sampler"]
        input_params = torch.repeat_interleave(true_params,batch_sizes[0],dim=0)
        theory_out = theory.forward(input_params)
        x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance = theory_out[:6]
        x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance = x_bins.cpu(), xsec_x.cpu(), Q2_bins.cpu(), xsec_Q2.cpu(), weights.cpu(), acceptance.cpu()
        x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance = x_bins.numpy(), xsec_x.numpy(), Q2_bins.numpy(), xsec_Q2.numpy(), weights.numpy(), acceptance.numpy()
        sycl_sampler.delay_recording()
        sycl_sampler.forward([x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance],n_gen_events)
        sycl_sampler.start_recording()

for i in range(1,1+n_scans):
    print(f"   Scan {i}/{n_scans}")
    batch_size = batch_sizes[i-1]
    input_params = torch.repeat_interleave(true_params,batch_size,dim=0)
    theory_out = theory.forward(input_params)
    dt_py = 0.0
    dt_cpp = 0.0
    dt_omp = 0.0
    dt_sycl = 0.0
    #++++++++++++++++++++++++
    for _ in range(n_trials):
        if RUN_PYTORCH:
            x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance = theory_out[:6]
            t_start = time.time()
            gen_events = sampler.forward([x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance],n_gen_events)
            t_end = time.time()
            df = pd.concat([df, pd.DataFrame({"host":hostname,"events":n_gen_events,"duration":[(t_end-t_start)],"implementation":"PyTorch"})])
            dt_py += ((t_end-t_start)/float(n_trials))
            #copy to the host for all future numerical checks
            gen_events = gen_events.cpu()
            #torch.set_num_threads(1)
            #if COLLECT_MEMORY_FOOTPRINT:
            #    max_memory_used = torch.cuda.max_memory_reserved()
            #    torch.cuda.reset_peak_memory_stats()
            #    with open('memory_footprints.csv', 'a') as the_file:
            #        the_file.write(str(n_gen_events)+','+str(max_memory_used)+'\n')
            #    print("Python reference implementation took: {} ".format(t_end - t_start))
            #    print("Max used GPU memory:", max_memory_used)

        if RUN_CPP:
            # Since we're done with the PyTorch stuff all data must be on the host CPU for our own wrappers
            x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance = theory_out[:6]
            x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance = x_bins.cpu(), xsec_x.cpu(), Q2_bins.cpu(), xsec_Q2.cpu(), weights.cpu(), acceptance.cpu()
            x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance = x_bins.numpy(), xsec_x.numpy(), Q2_bins.numpy(), xsec_Q2.numpy(), weights.numpy(), acceptance.numpy()
            ct_start = time.time()
            cpp_gen_events = cpp_sampler.forward([x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance],n_gen_events)
            ct_end = time.time()
            dt_cpp += ((ct_end-ct_start)/float(n_trials))
            #assert np.allclose(gen_events.numpy(), cpp_gen_events, atol=1e-2)
            df = pd.concat([df, pd.DataFrame({"host":hostname,"events":n_gen_events,"duration":[(ct_end-ct_start)],"implementation":"C++"})])
            print("C++ implementation took: {} ".format(ct_end - ct_start))

        if RUN_OMP:
            # Since we're done with the PyTorch stuff all data must be on the host CPU for our own wrappers
            x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance = theory_out[:6]
            x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance = x_bins.cpu(), xsec_x.cpu(), Q2_bins.cpu(), xsec_Q2.cpu(), weights.cpu(), acceptance.cpu()
            x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance = x_bins.numpy(), xsec_x.numpy(), Q2_bins.numpy(), xsec_Q2.numpy(), weights.numpy(), acceptance.numpy()
            ot_start = time.time()
            omp_gen_events = omp_sampler.forward([x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance],n_gen_events)
            ot_end = time.time()
            dt_omp += ((ot_end-ot_start)/float(n_trials))
            #assert np.allclose(gen_events.numpy(), omp_gen_events, atol=1e-2)
            df = pd.concat([df, pd.DataFrame({"host":hostname,"events":n_gen_events,"duration":[(ot_end-ot_start)],"implementation":"OpenMP"})])
            print("OpenMP implementation took: {} ".format(ot_end - ot_start))

        if RUN_SYCL:
            for sycl_sampler in sycl_samplers:
                sycl_implementation_name = sycl_sampler["name"]
                sycl_sampler = sycl_sampler["sampler"]
                # Since we're done with the PyTorch stuff all data must be on the host CPU for our own wrappers
                x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance = theory_out[:6]
                x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance = x_bins.cpu(), xsec_x.cpu(), Q2_bins.cpu(), xsec_Q2.cpu(), weights.cpu(), acceptance.cpu()
                x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance = x_bins.numpy(), xsec_x.numpy(), Q2_bins.numpy(), xsec_Q2.numpy(), weights.numpy(), acceptance.numpy()
                st_start = time.time()
                sycl_gen_events = sycl_sampler.forward([x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance],n_gen_events)
                st_end = time.time()
                dt_sycl += ((st_end-st_start)/float(n_trials))
                #if gen_events.numpy().shape != sycl_gen_events.shape:
                #    import ipdb; ipdb.set_trace()
                #assert np.allclose(gen_events.numpy(), sycl_gen_events, atol=1e-2)
                df = pd.concat([df, pd.DataFrame({"host":hostname,"events":n_gen_events,"duration":[(st_end-st_start)],"implementation":"SYCL-{}".format(sycl_implementation_name)})])
                print("SYCL implementation ({}) took: {} ".format(sycl_implementation_name,st_end - st_start))

    #++++++++++++++++++++++++
    if RUN_PYTORCH: py_avg_times.append(dt_py)
    if RUN_CPP: cpp_avg_times.append(dt_cpp)
    if RUN_OMP: omp_avg_times.append(dt_omp)
    if RUN_SYCL: sycl_avg_times.append(dt_sycl)
#+++++++++++++++++++++++++++
print("...done!")
print(" ")
df.to_csv('times.csv', mode='a', header=not os.path.exists("times.csv"))

print("Write results to file...")
print("batch sizes = {}".format(batch_sizes))
print("python average time = {}".format(py_avg_times))
if RUN_CPP: print("c++ average time = {}".format(cpp_avg_times))
if RUN_OMP: print("openmp average time = {}".format(omp_avg_times))
if RUN_SYCL: print("sycl average time = {}".format(sycl_avg_times))
np.save(logdir+"/batch_sizes.npy",np.array(batch_sizes))
np.save(logdir+"/python_avg_loits_times.npy",np.array(py_avg_times))

print("Freeing samplers...")
if RUN_PYTORCH:
    del sampler
if RUN_CPP:
    del cpp_sampler
if RUN_OMP:
    del omp_sampler
if RUN_SYCL:
    for smplr in sycl_samplers:
        smplr["sampler"].shutdown()
        del smplr["sampler"]

print("...done! Have a wonderful day!")
print(" ")
