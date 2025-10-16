import torch
from eco_5.graph import MultiLineGraph
import matplotlib.pyplot as plt
devc = torch.device(type="cuda")
torch.set_printoptions(sci_mode=False)


# -------- CONSTS --------
NUM_SIMS = 480
COUNT = 1
CHANCE_KEEP = .65
ITER_ALCHS = 300 # num of alchs sequentially
ITEM_COST = 10.4 # ave loss, in millions
ALCH_VALUE = 5.52 # ave win, m
"""
condensed gold: cost 10.4 -- val 5.52
magic stone: cost .975 -- val .5175
gold leaf: cost .13 -- val .069
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
    yScatter[a] = cash
    
    return cash
    

# -------- MAIN --------
xScatter = torch.arange(start=1, end=ITER_ALCHS + 1) # global scatter x axis
xScatter = xScatter.view([-1, 1]).repeat(repeats=[1, NUM_SIMS])

yScatter = torch.zeros([ITER_ALCHS, NUM_SIMS], device=devc, dtype=torch.float32)
"""
yScatter1 = torch.zeros_like(yScatter)
yScatter2 = torch.zeros_like(yScatter)
"""
#print(f"{yScatter}")

# cash starting
cash = torch.ones([NUM_SIMS])
cash.mul_(COUNT) # mul by num of start items
cash.mul_(ITEM_COST) # multiply starting value (total start value)
#print(f"{cash}")

# loop
for a in range(ITER_ALCHS):
    cash = highAlch(a, cash)



# cash stack scatter
ALPHA = .1
SIZE = 6
plt.scatter(
    xScatter.view([-1]).cpu(),
    yScatter.view([-1]).cpu(),
    c="red",
    alpha=ALPHA,
    s=SIZE
)

plt.legend(["3", "2", "1"])
plt.ylim((-20, 850))
plt.xlabel("num of alchs in sequence")
plt.ylabel("simulated cash stack")
plt.show()
