import torch
import concurrent.futures
from utils import logging
import numpy as np

profiler = False
#profiler = True

if profiler:
    import torch.cuda.nvtx as nvtx

torch.set_default_dtype(torch.float64)

class Torch2DInverseTransformSampler(object):
    """
    This class utilizes the code provided by Nobuo Sato to sample from a 2D grid, using the inverse CDF method.
    """

    # Initialize:
    # ***************************
    def __init__(self, config, devices="cpu", internal_logging=False):
        self.devices = devices

        self.vmap_randomness = config.get('vmap_randomness','different')
        self.use_threading = config.get('use_threading',False)
        self.requires_grad = config.get('requires_grad',True)
        self.gen_settings = {"x": [True, 1, 0, 0], "q2": [False, 0, 1, 1]}
        self.logger = None
        if internal_logging :
            self.logger = logging(internal_logging)
            self.logger.mark_for_recording("linear_interpolation")
            self.logger.mark_for_recording("calc_rho")
            self.logger.mark_for_recording("calc_cdf")
            self.logger.mark_for_recording("calc_grid_indices")
            self.logger.mark_for_recording("calc_weight_matrix")
            self.logger.mark_for_recording("uobs_fill_and_flattern");
            self.logger.mark_for_recording("bin_obs_flattern");
            self.logger.mark_for_recording("reshape");
            self.logger.mark_for_recording("filter");
            self.logger.mark_for_recording("forward_single_sample");
            self.logger.mark_for_recording("total");
    # ***************************

    def set_logger(self, logger):
        self.logger = logger

    # Components to calculate the CDF in x and Q2 coordinates:
    #***************************
    # Calculate the norm: Integral(xsec(x) dx), with: x = bins
    # which is approximated by the trapezoid rule:
    def calc_norm(self,bins,xsec):
        res = torch.trapezoid(xsec,bins)
        return res
    #--------------------------

    # Compute the weight matrix:
    # This piece of code was suggested by Steven Goldenberg --> This calculation is done on CPU --> We want to use the GPU for something else
    #***************************
    def calc_weight_tensor(self,n):
        if self.logger: self.logger.tic("calc_weight_matrix")
        res = n  > torch.arange(torch.max(n))
        if self.logger: self.logger.toc("calc_weight_matrix")
        return res
    #***************************

    # Compute grid indices so that we are able to vectorize the sampler
    # The following (very efficient) lines were written by Steven Goldenberg:
    #***************************
    def calc_grid_indices(self,size_dim_0,size_dim_1):
       if self.logger: self.logger.tic("calc_grid_indices")
       res = np.mgrid[0:size_dim_0,0:size_dim_1].reshape(2,-1).T
       if self.logger: self.logger.toc("calc_grid_indices")
       return res
    #***************************

    def slice_test(self):
        #proof of concept for c++ implementation to yield the same results
        data = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]
        tensor = torch.tensor(data)
        sum0 = torch.vmap(lambda s: sum(s),in_dims=0)(tensor)
        sum1 = torch.vmap(lambda s: sum(s),in_dims=1)(tensor)
        sum2 = torch.vmap(lambda s: sum(s),in_dims=2)(tensor)
        print("tensor = {}".format(tensor))
        print("sum along dim 0 = {}".format(sum0))
        print("sum along dim 1 = {}".format(sum1))
        print("sum along dim 2 = {}".format(sum2))

        data = torch.tensor([[[1, 2], [3, 4], [5, 6], [7, 8]],
                             [[9, 10], [11, 12], [13, 14], [15, 16]],
                             [[17, 18], [19, 20], [21, 22], [23, 24]]])
        print("tensor = {}, shape = {}".format(data,data.shape))
        sum0 = torch.vmap(lambda s: sum(s),in_dims=0)(data)
        sum1 = torch.vmap(lambda s: sum(s),in_dims=1)(data)
        sum2 = torch.vmap(lambda s: sum(s),in_dims=2)(data)
        print("sum along dim 0 = {}, shape = {}".format(sum0,sum0.shape))
        print("sum along dim 1 = {}, shape = {}".format(sum1,sum1.shape))
        print("sum along dim 2 = {}, shape = {}".format(sum2,sum2.shape))

    # Determine the density: rho = xsec / norm
    def calc_rho(self,bins,xsec):
        if self.logger: self.logger.tic("calc_rho")
        norm = torch.vmap(lambda s: self.calc_norm(bins,s),in_dims=0)(xsec)
        # and get rho:
        rho = xsec / norm[:,:,None]
        if self.logger: self.logger.toc("calc_rho")
        return rho

    # --------------------------

    # Calculate the inverse from the given CDF, by performing a linear interpolation from the
    # binned CDF from theory to the corresponding bin (in either x or Q2)
    #***************************
    # Run the linear interpolation:
    def linear_interpolation(self,u,cdf,bin):
        """
        This function is directly taken from Nobuo Satos original code
        The core function: bin(u) = m*u + b
        """

        if self.logger: self.logger.tic("linear_interpolation")
        if profiler: nvtx.range_push("linear_interpolation")
        # Determine m = (bin[i+1] - bin[i]) / (cdf[i+1] - cdf[i])
        m = (bin[1:] - bin[:-1]) / (cdf[1:] - cdf[:-1] + 1e-5)
        # b = -m*cdf[i] + bin[i]
        b = bin[:-1] - (m * cdf[:-1])
        # Make sure that we get the proper indices that obey: u >= cdf
        indicies = torch.sum(torch.ge(u[:, None], cdf[None, :]), 1) - 1
        # They should lie between 0 and n_points -1
        indicies = torch.clamp(indicies, 0,m.size()[0] - 1)
        # And return the interpolation:
        x = m[indicies] * u + b[indicies]
        if self.logger: self.logger.toc("linear_interpolation")
        if profiler: nvtx.range_pop()
        return x
    #***************************

    # Now compute the CDF itself:
    def calc_cdf(self,bins,xsec,acceptance,transpose_dim):
        rho = self.calc_rho(bins,xsec)
        if self.logger: self.logger.tic("calc_cdf")
        mult_acc = acceptance
        # Depending on whether we sample in the x or Q2 dimension,
        # we have to transpose the axis of the acceptance (for the x-dimension only)
        if transpose_dim:
            mult_acc = torch.transpose(acceptance, 0, 1)

        # Define an empty CDF tensor and overwrite it with the cumulative sum --> Steven Goldenberg worked on the lines below:
        cdf = torch.zeros_like(rho)
        cdf[:, :,1:] = torch.cumulative_trapezoid(
            y=rho, x=bins
        )
        cdf *= mult_acc[:, :, None]
        if self.logger: self.logger.toc("calc_cdf")
        return cdf
    # ***************************

    # Generate events, based on the provided x / Q2 grid, the acceptance matrix, weight tensors and grid index tensors:
    # ***************************
    # Define a helper function that generates one variable: (e.g. x or Q2)
    def generate_single_observable(
        self,
        obs_bins,
        obs_xsec,
        acceptance_matrix,
        weight_tensor,
        grid_index_tensor,
        n_max,
        obs_type,
    ):
        settings = self.gen_settings[obs_type]
        # Compute the CDF:
        cdf_obs = self.calc_cdf(obs_bins, obs_xsec, acceptance_matrix, settings[0])
        # First generate u:
        if self.logger: self.logger.tic("uobs_fill_and_flattern")
        #u_obs = torch.full((grid_index_tensor.size()[0], n_max),0.5)
        u_obs = torch.rand(
            (grid_index_tensor.size()[0], n_max),
            device=self.devices,
            dtype=torch.float32,
        )

        # Use the cdf in obs that we just computed and combine them with the index tensor,
        # i.e. assign the proper values to the grid position:
        cdf_obs_flat = cdf_obs[
                grid_index_tensor[:, settings[1]], grid_index_tensor[:, settings[2]]
            ]
        if self.logger: self.logger.toc("uobs_fill_and_flattern")

        # Do the same for the bins in x:
        if self.logger: self.logger.tic("bin_obs_flattern")
        bin_obs_flat = obs_bins[grid_index_tensor[:, settings[3]]]
        if self.logger: self.logger.toc("bin_obs_flattern")

        # Ensure all data is on the target device
        u_obs = u_obs.to(device=self.devices)
        cdf_obs_flat = cdf_obs_flat.to(device=self.devices)
        bin_obs_flat = bin_obs_flat.to(device=self.devices)
        weight_tensor = weight_tensor.to(device=self.devices)

        # Now utilize the linear interpolation function and compute x:
        # Also, we can now leverage that the indices are flat and run a vectorization:

        obs_gen = (
            torch.vmap(
                self.linear_interpolation, in_dims=0, randomness=self.vmap_randomness
            )(u_obs, cdf_obs_flat, bin_obs_flat)
            * weight_tensor
        )

        #simplified_obs_gen = torch.zeros((u_obs.shape[0],u_obs.shape[1]))
        #for p in range(u_obs.shape[0]):
        #    print("row = {}".format(p))
        #    simplified_obs_gen[p] = self.tmp_linear_interpolation(u_obs[p],cdf_obs_flat[p], bin_obs_flat[p])
        #simplified_obs_gen = simplified_obs_gen*weight_tensor
        #print(obs_gen.shape)
        #assert torch.eq(simplified_obs_gen,obs_gen).all()
        if self.logger: self.logger.tic("reshape")
        res = obs_gen.flatten()[:, None]
        if self.logger: self.logger.toc("reshape")
        return res

    # --------------------------------------------

    def gen_events(
        self,
        x_bins,
        xsec_x,
        Q2_bins,
        xsec_Q2,
        acceptance_matrix,
        weight_tensor,
        grid_index_tensor,
        n_max,
    ):
        x_gen = self.generate_single_observable(
            x_bins,
            xsec_x,
            acceptance_matrix,
            weight_tensor,
            grid_index_tensor,
            n_max,
            "x",
        )
        Q2_gen = self.generate_single_observable(
            Q2_bins,
            xsec_Q2,
            acceptance_matrix,
            weight_tensor,
            grid_index_tensor,
            n_max,
            "q2",
        )
        assert x_gen.numel() > 0
        assert Q2_gen.numel() > 0
        return torch.cat([x_gen, Q2_gen], dim=1)
    # ***************************

    # Define a forward pass:
    # ***************************
    # Forward pass for a single parameter sample:
    def forward_single_sample(self,x_bins,xsec_x,Q2_bins,xsec_Q2,acceptance,weight_tensor,grid_idx,max_n):
        if self.devices == "cuda":
            x_bins        = x_bins.to(device='cuda')
            xsec_x        = xsec_x.to(device='cuda')
            Q2_bins       = Q2_bins.to(device='cuda')
            xsec_Q2       = xsec_Q2.to(device='cuda')
            acceptance    = acceptance.to(device='cuda')
            weight_tensor = weight_tensor.to(device='cuda')
            grid_idx      = grid_idx.to(device='cuda')
            max_n         = max_n.to(device='cuda')

        # Generate events:
        evts = self.gen_events(
            x_bins, xsec_x, Q2_bins, xsec_Q2, acceptance, weight_tensor, grid_idx, max_n
        )
        if self.logger: self.logger.tic("filter")
        if profiler: nvtx.range_push("filter")
        cond = evts[:, 0] * evts[:, 1] != 0
        events = evts[cond]

        cond2 = (~torch.isnan(events[:, 0])) & (~torch.isnan(events[:, 1]))
        events = events[cond2]
        if profiler: nvtx.range_pop()

        if self.logger: self.logger.toc("filter")

        # Detach everything that we do not need --> Free the GPU memory:
        del cond
        del evts
        del cond2

        # Free cache on the device we are using
        #empty_cache(self.devices)

        return events

    # --------------------------------------------

    # Formulate the entire forward pass:
    def forward(self, theory_outputs, n_events):
        if self.logger: self.logger.tic("total")

        # Get the information from theory: We are expecting 6 items:
        # i) bins and crossections in x and Q^2 coordinates = 4 variables
        # ii) cross section weights and phase space acceptance = 2 variables
        x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance = theory_outputs[:6]

        if self.requires_grad:
            x_bins.requires_grad  = True
            xsec_x.requires_grad  = True
            Q2_bins.requires_grad = True
            xsec_Q2.requires_grad = True
            weights.requires_grad = True

        # Compute flat grid-indices which help to avoid another for-loop over the grid itself:
        npy_grid_idx = self.calc_grid_indices(weights[0].size()[0],weights[0].size()[1])
        torch_grid_idx = torch.as_tensor(npy_grid_idx,device=self.devices,dtype=torch.int)

        # Get batch size from theory outputs:
        batch_size = x_bins.size()[0]
        data_list = []

        # Generate data, using one thread per prediction, where #events_generated = n_events
        if self.use_threading == True:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=batch_size
            ) as executor:
                threads = []

                # +++++++++++++++++++++++++++++++
                for i in range(batch_size):
                    n_low = i * n_events
                    n_high = (i + 1) * n_events
                    # Run this on CPU:
                    detached_weights = weights[i].detach().cpu()
                    n = torch.abs(detached_weights * n_high).to(torch.int).reshape(
                        -1, 1
                    ) - torch.abs(detached_weights * n_low).to(torch.int).reshape(-1, 1)
                    max_n = torch.max(n)

                    weight_tensor_cpu = self.calc_weight_tensor(n)
                    weight_tensor_gpu = torch.as_tensor(
                        weight_tensor_cpu, device=self.devices, dtype=torch.float32
                    )
                    if self.logger: self.logger.tic("forward_single_sample")
                    threads.append(
                        executor.submit(
                            self.forward_single_sample,
                            x_bins[i],
                            xsec_x[i],
                            Q2_bins[i],
                            xsec_Q2[i],
                            acceptance[i],
                            weight_tensor_gpu,
                            torch_grid_idx,
                            max_n,
                        )
                    )
                    if self.logger: self.logger.toc("forward_single_sample")

                # +++++++++++++++++++++++++++++++

                # +++++++++++++++++++++++++++++++
                for thread in concurrent.futures.as_completed(threads):
                    data_list.append(thread.result())
                # +++++++++++++++++++++++++++++++

        else:
            # Or run everything sequentially and generate n_events for each prediction, where #events_generated = batch_size * n_events
            # +++++++++++++++++++++++++++++++
            for i in range(batch_size):
                  # Run this on CPU:
                  detached_weights = weights[i].detach().cpu()
                  #new way:
                  #w_sum = torch.zeros(
                  #      detached_weights.flatten().shape[0] + 1,
                  #      device="cpu"
                  #  )
                  #w_sum[1:] = torch.cumsum(detached_weights.flatten(), 0)
                  #u = torch.rand(n_events, device="cpu")
                  #n = (
                  #    torch.histogram(u, bins=w_sum)
                  #    .hist.to(torch.int)
                  #    .reshape(-1, 1)
                  #)
                  #vs
                  #old way:
                  n = torch.abs(detached_weights*n_events).to(torch.int).reshape(-1,1)
                  max_n = torch.max(n)

                  weight_tensor_cpu = self.calc_weight_tensor(n)
                  weight_tensor_gpu = torch.as_tensor(weight_tensor_cpu,device=self.devices,dtype=torch.float32)
                  if self.logger: self.logger.tic("forward_single_sample")
                  data_list.append(self.forward_single_sample(x_bins[i], xsec_x[i], Q2_bins[i], xsec_Q2[i],acceptance[i],weight_tensor_gpu,torch_grid_idx,max_n))
                  if self.logger: self.logger.toc("forward_single_sample")
            #+++++++++++++++++++++++++++++++
        res = torch.cat(data_list,dim=0)
        if self.logger: self.logger.toc("total")

        return res
    #***************************

    # Apply
    #***************************
    def apply(self,theory_outputs,n_events):
        return self.forward(theory_outputs,n_events)
    #***************************
