
class logging:

    _filename = "times.csv"
    _timings = {}
    _record_list = []
    _file_output_stream = None

    def __init__(self,filename):
        self.assign_filename_for_recording(filename)
        self._file_output_stream = open(filename,"a")

    def assign_filename_for_recording(self,filename):
        self._filename = filename

    def mark_for_recording(self,region_name):
        self._record_list.append(region_name)

    def tic(self, region_name, mode=0):
        import time
        now = time.time_ns();
        if mode == 0:
            self._timings[region_name] = now
        else:
            total_time = now - self._timings[region_name]
            print("{} : {} seconds\n".format(region_name,total_time*1E-9))
            #and record it
            is_present = region_name in self._record_list
            if is_present:
                self._file_output_stream.write("{},{}\n".format(region_name,total_time*1E-9))

    def toc(self,region_name):
        self.tic(region_name,1)

