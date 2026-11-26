"""Fail before loading Wan if the rank's corresponding HBM NUMA cannot be bound."""
import ctypes
import json
import mmap
import os
from pathlib import Path
import socket
import sys

rank=int(os.environ['OMPI_COMM_WORLD_RANK'])
local_rank=int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
node=16+local_rank
size=2*1024*1024
mapped=mmap.mmap(-1,size,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS,
                 prot=mmap.PROT_READ|mmap.PROT_WRITE)
libc=ctypes.CDLL("libnuma.so.1",use_errno=True)
libc.mbind.argtypes=[ctypes.c_void_p,ctypes.c_ulong,ctypes.c_int,
                     ctypes.POINTER(ctypes.c_ulong),ctypes.c_ulong,ctypes.c_uint]
libc.mbind.restype=ctypes.c_int
mask=ctypes.c_ulong(1<<node)
ptr=ctypes.addressof(ctypes.c_char.from_buffer(mapped))
rc=libc.mbind(ctypes.c_void_p(ptr),ctypes.c_ulong(size),2,
              ctypes.byref(mask),ctypes.c_ulong(64),ctypes.c_uint(0))
err=ctypes.get_errno() if rc else 0
record={'host':socket.gethostname(),'rank':rank,'local_rank':local_rank,
        'expected_hbm_numa':node,'mbind_rc':rc,'errno':err,
        'mems_allowed_list':next((line.split(':',1)[1].strip() for line in
            Path('/proc/self/status').read_text().splitlines()
            if line.startswith('Mems_allowed_list:')),None),
        'hbm_node_online':Path(f'/sys/devices/system/node/node{node}').is_dir(),
        'cpu_affinity_size':len(os.sched_getaffinity(0))}
print('WAN21_HBM_BIND_PROBE '+json.dumps(record,sort_keys=True),flush=True)
if local_rank == 0:
    devices=[]
    for nic in Path('/sys/class/infiniband').glob('*'):
        numa=nic/'device'/'numa_node'
        ports=[]
        for port in (nic/'ports').glob('*'):
            state=port/'state'
            ports.append({'port':port.name,'state':state.read_text().strip() if state.is_file() else None})
        devices.append({'device':nic.name,
                        'numa_node':numa.read_text().strip() if numa.is_file() else None,
                        'ports':ports})
    print('WAN21_NIC_TOPOLOGY '+json.dumps({'host':socket.gethostname(),
           'devices':sorted(devices,key=lambda device:device['device'])}),flush=True)
if os.environ.get('WAN21_SHARED_HBM_PROBE', '0') == '1' and local_rank == 0:
    import tempfile
    with tempfile.NamedTemporaryFile(prefix='wan21_hbm_shared_probe_',
                                     dir='/tmp') as temporary:
        os.ftruncate(temporary.fileno(), size)
        shared=mmap.mmap(temporary.fileno(), size, flags=mmap.MAP_SHARED,
                         prot=mmap.PROT_READ|mmap.PROT_WRITE)
        address=ctypes.addressof(ctypes.c_char.from_buffer(shared))
        shared_rc=libc.mbind(ctypes.c_void_p(address),ctypes.c_ulong(size),2,
                             ctypes.byref(mask),ctypes.c_ulong(64),ctypes.c_uint(0))
        shared_err=ctypes.get_errno() if shared_rc else 0
        if not shared_rc:
            shared[0]=1
        numa_line=next((line for line in Path('/proc/self/numa_maps').read_text().splitlines()
                        if line.split()[0] == format(address, 'x')), None)
        print('WAN21_SHARED_HBM_BIND_PROBE '+json.dumps({
            'host':socket.gethostname(),'rank':rank,'node':node,
            'mbind_rc':shared_rc,'errno':shared_err,'numa_map_line':numa_line}),flush=True)
        shared.close()
        if shared_rc or not numa_line or f'N{node}=' not in numa_line:
            sys.exit(87)
mapped.close()
if rc:
    sys.exit(86)
