import os
import builtins
import numpy as np
import torch
from torch.autograd import Function
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
try:
    if torch.__version__[:6] >= '1.12.0':
        import oneccl_bindings_for_pytorch
    else:
        import torch_ccl
except ImportError as e:
    #print(e)
    oneccl_bindings_for_pytorch = False

my_rank = -1
my_size = -1
my_local_rank = -1
my_local_size = -1
alltoall_supported = False
allgatherv_supported = False
a2a_impl = os.environ.get('DLRM_ALLTOALL_IMPL', '')

myreq = None

def env2int(env_list, default = -1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0: return val
    return default

def get_my_slice(n):
    my_size = dist.get_world_size()
    my_rank = dist.get_rank()
    k, m = divmod(n, my_size)
    return slice(my_rank * k + min(my_rank, m), (my_rank+1) * k + min(my_rank+1, m), 1)

def get_split_lengths(n):
    my_size = dist.get_world_size()
    k, m = divmod(n, my_size)
    if m == 0:
        splits = None
        my_len = k
    else:
        my_rank = dist.get_rank()
        splits = [(k+1) if i < m else k for i in range(my_size)]
        my_len = splits[my_rank]
    return (my_len, splits)

def init_distributed(rank = -1, size = -1, backend=''):
    global myreq
    #global my_rank
    global my_size
    global my_local_rank
    global my_local_size
    global a2a_impl
    global alltoall_supported
    global allgatherv_supported
    # guess MPI ranks from env (works for IMPI, OMPI and MVAPICH2)
    num_mpi_ranks = env2int(['PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE', 'WORLD_SIZE'])
    if backend == '' and num_mpi_ranks > 1:
        if oneccl_bindings_for_pytorch and env2int(['CCL_WORKER_COUNT']) > 0:
            backend = 'ccl'
        elif dist.is_mpi_available():
            backend = 'mpi'
        else:
            print("WARNING: MPI multi-process launch detected but PyTorch MPI backend not available.")
            backend = 'gloo'
    if backend != '':
        #guess Rank and size
        if rank == -1:
            rank = env2int(['PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'MV2_COMM_WORLD_RANK', 'RANK'], 0)
        if size == -1:
            size = env2int(['PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE', 'WORLD_SIZE'], 1)
        if not os.environ.get('RANK', None) and rank != -1: os.environ['RANK'] = str(rank)
        if not os.environ.get('WORLD_SIZE', None) and size != -1: os.environ['WORLD_SIZE'] = str(size)
        if not os.environ.get('MASTER_PORT', None): os.environ['MASTER_PORT'] = '29500'
        if not os.environ.get('MASTER_ADDR', None):
            local_size = env2int(['MPI_LOCALNRANKS', 'OMPI_COMM_WORLD_LOCAL_SIZE', 'MV2_COMM_WORLD_LOCAL_SIZE'], 1)
            if local_size != size and backend != 'mpi':
                print("Warning: Looks like distributed multinode run but MASTER_ADDR env not set, using '127.0.0.1' as default")
                print("If this run hangs, try exporting rank 0's hostname as MASTER_ADDR")
            os.environ['MASTER_ADDR'] = '127.0.0.1'
    if size > 1:
        dist.init_process_group(backend, rank=rank, world_size=size)
        my_rank = dist.get_rank()
        my_size = dist.get_world_size()
        my_local_rank = env2int(['MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'MV2_COMM_WORLD_LOCAL_RANK'], 0)
        my_local_size = env2int(['MPI_LOCALNRANKS', 'OMPI_COMM_WORLD_LOCAL_SIZE', 'MV2_COMM_WORLD_LOCAL_SIZE'], 1)
        if my_rank == 0: print("Running on %d ranks using %s backend" % (my_size, backend))
        if backend == 'ccl':
            print("Using CCL_ATL_TRANSPORT=%s" % os.environ.get('CCL_ATL_TRANSPORT', '(default)'))
            print("Using CCL_ATL_SHM=%s" % os.environ.get('CCL_ATL_SHM', '(default)'))
        if hasattr(dist, 'all_to_all_single'):
            try:
               # dist.all_to_all_single(torch.empty([0]), torch.empty([0]))
                alltoall_supported = True
            except RuntimeError:
                pass
        if a2a_impl == 'alltoall' and alltoall_supported == False:
            print("Requested DLRM_ALLTOALL_IMPL=%s but backend %s does not support it, use scatter/gather based alltoall" % (a2a_impl, backend))
            a2a_impl = 'scatter'
        if a2a_impl != '': print("Using DLRM_ALLTOALL_IMPL=%s" % a2a_impl)
        try:
            x = torch.ones([my_rank])
            y = torch.zeros([(my_size*(my_size-1))//2])
            y = list(y.split([r for r in range(my_size)]))
            dist.all_gather(y, x)
            allgatherv_supported = True
        except RuntimeError:
            pass
    else:
        my_rank = 0
        my_size = 1
        my_local_rank = 0
        my_local_size = 1
    myreq = Request()

class Request(object):
    def __init__(self):
        self.req = None
        self.tensor = None
        self.WaitFunction = All2All_Scatter_Wait

    def wait(self):
        ret = self.WaitFunction.apply(*self.tensor)
        self.req = None
        self.tensor = None
        return ret

class All2All_ScatterList_Req(Function):
    @staticmethod
    def forward(ctx, a2ai, *inputs):
        global myreq
        my_rank = dist.get_rank()
        #print("All2All_ScatterList_Req:forward")
        mb_split_lengths = a2ai.gNS if a2ai.gNS else a2ai.lN
        emb_split_lengths = a2ai.gSS if a2ai.gSS else [a2ai.lS] * my_size
        gather_list = []
        req_list = []
        for i in range(my_size):
            for j in range(emb_split_lengths[i]):
                out_tensor = inputs[0].new_empty([a2ai.lN, a2ai.E])
                scatter_list = list(inputs[j].split(mb_split_lengths, dim = 0)) if i == my_rank else []
                req = dist.scatter(out_tensor, scatter_list, src=i, async_op=True)
                gather_list.append(out_tensor)
                req_list.append(req)
        myreq.req = req_list
        myreq.tensor = tuple(gather_list)
        myreq.a2ai = a2ai
        return myreq.tensor

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        #print("All2All_ScatterList_Req:backward")
        for r in myreq.req:
            r.wait()
        myreq.req = None
        grad_inputs = myreq.tensor
        myreq.tensor = None
        return (None, *grad_inputs)


class All2All_ScatterList_Wait(Function):
    @staticmethod
    def forward(ctx, *output):
        global myreq
        #print("All2All_Scatter_Wait:forward")
        ctx.a2ai = myreq.a2ai
        for r in myreq.req:
            r.wait()
        myreq.req = None
        myreq.tensor = None
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        my_rank = dist.get_rank()
        a2ai = ctx.a2ai
        grad_output = [t.contiguous() for t in grad_output]
        mb_split_lengths = a2ai.gNS if a2ai.gNS else [a2ai.lN] * my_size
        per_rank_split_lengths = a2ai.gSS if a2ai.gSS else [a2ai.lS] * my_size
        grad_inputs = [grad_output[0].new_empty([ctx.a2ai.N, ctx.a2ai.E]) for _ in range(a2ai.lS)]
        req_list = []
        ind = 0
        for i in range(my_size):
            for j in range(per_rank_split_lengths[i]):
                gather_list = list(grad_inputs[j].split(mb_split_lengths, dim = 0)) if i == my_rank else None
                req = dist.gather(grad_output[ind], gather_list, dst = i, async_op=True)
                req_list.append(req)
                ind += 1
        myreq.req = req_list
        myreq.tensor = grad_inputs
        return tuple(grad_output)



class All2All_Scatter_Req(Function):
    @staticmethod
    def forward(ctx, a2ai, *inputs):
        global myreq
        #print("All2All_Scatter_Req:forward")
        my_rank = dist.get_rank()
        mb_split_lengths = a2ai.gNS if a2ai.gNS else a2ai.lN
        emb_split_lengths = a2ai.gSS if a2ai.gSS else [a2ai.lS] * my_size
        input = torch.cat(inputs, dim=1)
        scatter_list = list(input.split(mb_split_lengths, dim=0))
        gather_list = []
        req_list = []
        for i in range(my_size):
            out_tensor = input.new_empty([a2ai.lN, emb_split_lengths[i] * a2ai.E])
            req = dist.scatter(out_tensor, scatter_list if i == my_rank else [], src=i, async_op=True)
            gather_list.append(out_tensor)
            req_list.append(req)
        myreq.req = req_list
        myreq.tensor = tuple(gather_list)
        myreq.a2ai = a2ai
        ctx.a2ai = a2ai
        return myreq.tensor

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        #print("All2All_Scatter_Req:backward")
        for r in myreq.req:
            r.wait()
        myreq.req = None
        grad_input = myreq.tensor
        grad_inputs = grad_input.split(ctx.a2ai.E, dim=1)
        myreq.tensor = None
        return (None, *grad_inputs)


class All2All_Scatter_Wait(Function):
    @staticmethod
    def forward(ctx, *output):
        global myreq
        #print("All2All_Scatter_Wait:forward")
        ctx.a2ai = myreq.a2ai
        for r in myreq.req:
            r.wait()
        myreq.req = None
        myreq.tensor = None
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        my_rank = dist.get_rank()
        #print("All2All_Scatter_Wait:backward")
        assert len(grad_output) == my_size
        scatter_list = [t.contiguous() for t in grad_output]
        a2ai = ctx.a2ai
        mb_split_lengths = a2ai.gNS if a2ai.gNS else a2ai.lN
        emb_split_lengths = a2ai.gSS if a2ai.gSS else [a2ai.lS] * my_size
        grad_input = grad_output[0].new_empty([a2ai.N, a2ai.E*a2ai.lS])
        gather_list = list(grad_input.split(mb_split_lengths, dim=0))
        req_list = []
        for i in range(my_size):
            #req = dist.scatter(gather_list[i], scatter_list if i == my_rank else [], src=i, async_op=True)
            req = dist.gather(scatter_list[i], gather_list if i == my_rank else [], dst=i, async_op=True)
            req_list.append(req)
        myreq.req = req_list
        myreq.tensor = grad_input
        return grad_output


class All2All_Req(Function):
    @staticmethod
    def forward(ctx, a2ai, *inputs):
        global myreq
        #print("All2All_Req:forward")
        mb_split_lengths = a2ai.gNS
        if mb_split_lengths: mb_split_lengths = [m * a2ai.lS * a2ai.E for m in mb_split_lengths]
        emb_split_lengths = a2ai.gSS
        if emb_split_lengths: emb_split_lengths = [a2ai.lN * e * a2ai.E for e in emb_split_lengths]
        input = torch.cat(inputs, dim=1).view([-1])
        output = input.new_empty([a2ai.S*a2ai.lN*a2ai.E])
        req = dist.all_to_all_single(output, input, emb_split_lengths, mb_split_lengths, async_op=True)
        myreq.req = req
        myreq.tensor = []
        myreq.tensor.append(output)
        myreq.tensor = tuple(myreq.tensor)
        a2ai.mb_split_lengths = mb_split_lengths
        a2ai.emb_split_lengths = emb_split_lengths
        myreq.a2ai = a2ai
        ctx.a2ai = a2ai
        return myreq.tensor

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        #print("All2All_Req:backward")
        a2ai = ctx.a2ai
        myreq.req.wait()
        myreq.req = None
        grad_input = myreq.tensor
        grad_inputs = grad_input.view([a2ai.N, -1]).split(a2ai.E, dim=1)
        grad_inputs = [gin.contiguous() for gin in grad_inputs]
        myreq.tensor = None
        return (None, *grad_inputs)


class All2All_Wait(Function):
    @staticmethod
    def forward(ctx, *output):
        global myreq
        #print("All2All_Wait:forward")
        a2ai = myreq.a2ai
        ctx.a2ai = a2ai
        myreq.req.wait()
        myreq.req = None
        myreq.tensor = None
        emb_split_lengths = a2ai.emb_split_lengths if a2ai.emb_split_lengths else a2ai.lS * a2ai.lN * a2ai.E
        outputs = output[0].split(emb_split_lengths)
        outputs = tuple([out.view([a2ai.lN, -1]) for out in outputs])
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        global myreq
        #print("All2All_Wait:backward")
        a2ai = ctx.a2ai
        grad_outputs = [gout.contiguous().view([-1]) for gout in grad_outputs]
        grad_output = torch.cat(grad_outputs)
        grad_input = grad_output.new_empty([a2ai.N * a2ai.lS * a2ai.E])
        req = dist.all_to_all_single(grad_input, grad_output, a2ai.mb_split_lengths, a2ai.emb_split_lengths, async_op=True)
        myreq.req = req
        myreq.tensor = grad_input
        return (grad_output,)

class AllGather(Function):

    @staticmethod
    def forward(ctx, input, global_lengths, dim=0):
        if not isinstance(global_lengths, (list, tuple)):
            global_lengths = [global_lengths] * my_size
        my_rank = dist.get_rank()
        assert(len(global_lengths) == my_size)
        assert(global_lengths[my_rank] == input.size(dim))
        local_start = sum(global_lengths[:my_rank])

        output_size = list(input.size())

        ctx.dim = dim
        ctx.local_start = local_start
        ctx.local_length = global_lengths[my_rank]

        input = input.contiguous()
        if dim == 0:
            out_len = sum(global_lengths)
            output_size[dim] = out_len
            output = input.new_empty(output_size)
            gather_list = list(output.split(global_lengths, dim=0))
        else:
            gather_list = [torch.empty_like(input) for _ in range(my_size)]
            gather_list = []
            for l in global_lengths:
                output_size[dim] = l
                gather_list.append(input.new_empty(output_size))

        dist.all_gather(gather_list, input)

        if dim != 0:
            output = torch.cat(gather_list, dim=dim)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # print("Inside All2AllBackward")
        dim = ctx.dim
        start = ctx.local_start
        length = ctx.local_length

        grad_input = grad_output.narrow(dim, start, length)

        return (grad_input, None, None)

class All2AllInfo(object):
    pass

def alltoall(inputs, per_rank_split_lengths):
    global myreq
    N, E = inputs[0].size()
    a2ai = All2AllInfo()
    a2ai.lS = len(inputs)
    a2ai.gSS = per_rank_split_lengths
    a2ai.lN, a2ai.gNS = get_split_lengths(N)
    a2ai.E = E
    a2ai.N = N
    a2ai.S = sum(per_rank_split_lengths) if per_rank_split_lengths else a2ai.lS * my_size
    if a2a_impl == '' and alltoall_supported or a2a_impl == 'alltoall':
        output = All2All_Req.apply(a2ai, *inputs)
        myreq.WaitFunction = All2All_Wait
    elif a2a_impl == '' or a2a_impl == 'scatter':
        #print("Using All2All_Scatter_Req")
        output = All2All_Scatter_Req.apply(a2ai, *inputs)
        myreq.WaitFunction = All2All_Scatter_Wait
    elif a2a_impl == 'scatter_list':
        #print("Using All2All_ScatterList_Req")
        output = All2All_ScatterList_Req.apply(a2ai, *inputs)
        myreq.WaitFunction = All2All_ScatterList_Wait
    else:
        print("Unknown value set for DLRM_ALLTOALL_IMPL (%s), please use one of [alltoall, scatter, scatter_list]" % a2a_impl) 
    return myreq

def shuffle_data(inputs):
    input = torch.cat(inputs)
    output = input.new_empty(input.size())
    req = dist.all_to_all_single(output, input) 
    output = output.reshape(my_size, -1)
    return output
    

def all_gather(input, lengths, dim=0):
    #print("lengths: ", lengths)
    if not lengths: lengths = [input.size(0)] * my_size
    return AllGather.apply(input, lengths, dim)

def all_gather_validation(input, lengths, dim=0):
    #print("lengths: ", lengths)
    if not lengths: lengths = [input.size(0)] * my_size

    global_lengths = lengths

    if not isinstance(global_lengths, (list, tuple)):
        global_lengths = [global_lengths] * my_size
    my_rank = dist.get_rank()
    assert(len(global_lengths) == my_size)
    assert(global_lengths[my_rank] == input.size(dim))
    local_start = sum(global_lengths[:my_rank])

    output_size = list(input.size())

    input = input.contiguous()
    if dim == 0:
        out_len = sum(global_lengths)
        output_size[dim] = out_len
        output = input.new_empty(output_size)
        gather_list = list(output.split(global_lengths, dim=0))
    else:
        gather_list = [torch.empty_like(input) for _ in range(my_size)]
        gather_list = []
        for l in global_lengths:
            output_size[dim] = l
            gather_list.append(input.new_empty(output_size))

    # if dim != 0:
    #     output = torch.cat(gather_list, dim=dim)
    req = dist.all_gather(gather_list, input, async_op=True)
    return req, output

def barrier():
    if my_size > 1:
        dist.barrier()

