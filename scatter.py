import torch
from eco_5.graph import MultiLineGraph
import matplotlib.pyplot as plt
devc = torch.device(type="cuda")
torch.set_printoptions(sci_mode=False)


# -------- CONSTS --------
NUM_SIMS = 240
START_COUNT = 4
CHANCE_KEEP = .65
ITER_ALCHS = 500 # num of alchs sequentially
ITEM_COST = 10.4 # ave loss, in millions
ALCH_VALUE = 5.52 # ave win, m
"""
condensed gold: cost 10.4 -- val 5.52
magic stone: cost .975 -- val .5175
gold leaf: cost .13 -- val .69
"""


# -------- HIGH ALCH --------
def highAlch(a, cash):
    # chance to keep, 1=keep, 0=toss
    roll = torch.rand_like(cash) # 0.0 - <1.0
    #print(f"roll: {roll}")

    # logic, keep +ALCH_VALUE, lose ALCH_VALUE-ITEM_COST
    change = torch.where(roll <= .65, ALCH_VALUE, ALCH_VALUE-ITEM_COST)
    
    # check gamblers ruin (if it has 1 item), 1=continue, 0=gamblers ruin
    keepGoing = torch.where(cash >= ITEM_COST, 1, 0)
    #print(f"go: {keepGoing}")
    
    # modify change
    change.mul_(keepGoing)
    #print(f"cva: {change}")
    
    # add change
    cash.add_(change)
    #print(f"new cash: {cash}\n")
    
    # track cash over alchs(time)
    yScatter0[a] = cash[:, 0] # 0 refers to 0th numStartItems which is 1 item
    yScatter1[a] = cash[:, 1]
    yScatter2[a] = cash[:, 2]
    """
    """
    
    return cash
    

# -------- MAIN --------
xScatter = torch.arange(start=1, end=ITER_ALCHS + 1) # global scatter x axis
xScatter = xScatter.view([-1, 1]).repeat(repeats=[1, NUM_SIMS])

yScatter0 = torch.zeros([ITER_ALCHS, NUM_SIMS], device=devc, dtype=torch.float32)
yScatter1 = torch.zeros_like(yScatter0)
yScatter2 = torch.zeros_like(yScatter0)
"""
"""
#print(f"{yScatter0}")

# cash starting
"""
cash = torch.arange( # range [START_COUNT]
    start=7, # 1
    end=8, # START_COUNT + 1
    device=devc, 
    dtype=torch.float32
)
"""
cash = torch.tensor([1, 3, 5], device=devc, dtype=torch.float32)
cash = cash.view([1, -1]).repeat(repeats=[NUM_SIMS, 1]) # y:multiple sims [NUM_SIMS, START_COUNT]
cash.mul_(ITEM_COST) # multiply starting value (total start value)
#print(f"{cash}")

# loop
for a in range(ITER_ALCHS):
    cash = highAlch(a, cash)



# cash stack scatter
ALPHA = 1
SIZE = 1
"""
print(f"{xScatter}")
print(f"{yScatter0}")
"""
# jitter l/r
xScatter2 = torch.add(xScatter, .2)
xScatter1 = torch.add(xScatter, -.2)

# 2 num
plt.scatter(
    xScatter2.view([-1]).cpu(),
    yScatter2.view([-1]).cpu(),
    c="aquamarine",
    alpha=ALPHA,
    s=SIZE
)
# 1 num
plt.scatter(
    xScatter1.view([-1]).cpu(),
    yScatter1.view([-1]).cpu(),
    c="dodgerblue",
    alpha=ALPHA,
    s=SIZE
)
"""
"""
# 0 num
plt.scatter(
    xScatter.view([-1]).cpu(),
    yScatter0.view([-1]).cpu(),
    c="red",
    alpha=ALPHA,
    s=SIZE
)

plt.legend(["3", "2", "1"])
plt.ylim((0.0, 800))
plt.xlabel("num of alchs in sequence")
plt.ylabel("simulated cash stack")
plt.show()
