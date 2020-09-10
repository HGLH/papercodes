from tensorboardX import SummaryWriter
s="dfe"
writer = SummaryWriter('runs')
for i in range(10):
    writer.add_scalar('quadratic', i**2, global_step=i)
    writer.add_scalar('exponential', 2**i, global_step=i)
#tensorboard --logdir runs