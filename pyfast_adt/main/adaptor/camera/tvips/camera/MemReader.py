import logging
import mmap
import numpy as np # type: ignore
from tvips.common.types import Axes2D

LOG = logging.getLogger(__name__)

class MemReader:
    dt_camc_shared_params = np.dtype([
        ('buf_index', np.uint32),
        ('total_buf_size', np.uint32),
        ('num_buf', np.uint32),
        ('offset_hist', np.uint32),
        ('offset_image16', np.uint32),
        ('offset_image8', np.uint32),
        ('offset_min', np.uint32),
        ('offset_max', np.uint32),
        ('offset_mean', np.uint32),
        ('offset_std', np.uint32),
        ('offset_power', np.uint32),
        ('offset_linescan', np.uint32),
        ('offset_plugin_data', np.uint32),
        ('plugin0_GUID', np.byte, 16),
        ('plugin0_offset', np.uint32), #relative to offset_plugin_data
        ('plugin0_data_size', np.uint32),
        ('reserved', np.byte, 52) #more plugin data here
        ])

    def _ReadSharedParams(self):
        params_mmap = mmap.mmap(-1, 128, "CAMC_SHARED_PARAMS_BUFFER")
        params_mmap.seek(0)
        params = np.frombuffer(params_mmap.read(128), self.dt_camc_shared_params)
        params_mmap.close()

        LOG.debug(params.__repr__())

        return params
        
    def GetLastImage(self, dimension: Axes2D) -> np.ndarray:
        params = self._ReadSharedParams()
        idx = params['buf_index'][0]
        bufsize = params['total_buf_size'][0]
        numbuf = params['num_buf'][0]
        LOG.debug("Reading out last image: Index {}".format(idx))
        
        #use that information for opening camc livebuffer
        LOG.debug("opening file with {:d} bytes, that's {:d} Mb".format(bufsize * numbuf, int(bufsize * numbuf / 1024 / 1024)))
        livebuffer = mmap.mmap(-1, bufsize * numbuf, "CAMC_SHARED_LIVEBUFFER") #Note: if that call fails, you probably run a 32 bit python.
        livebuffer.seek(idx * bufsize + params['offset_image16'][0])
        image = np.frombuffer(livebuffer.read(2 * dimension.x * dimension.y), dtype=np.uint16) #todo: check whether this is a copy or just a view of the memory
        LOG.debug("livebuffer read")
        image.shape = (dimension.x, dimension.y)
        livebuffer.close()
        
        LOG.debug("returning image")
        return image

