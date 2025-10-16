import torch
from eco_5.graph import MultiLineGraph
devc = torch.device(type="cuda")
torch.set_printoptions(sci_mode=False)


# -------- CONSTS --------
NUM_SIMS = 120000 # sims ran in parallel
ITER_ALCHS = 20000 # num of alchs sequentially
START_COUNT = 7
CHANCE_KEEP = .65
ITEM_COST = .13 # ave loss, in millions
ALCH_VALUE = .069 # ave win, m
"""
condensed gold: cost 10.4 -- val 5.52
magic stone: cost .975 -- val .5175
gold leaf: cost .13 -- val .069
"""


# -------- HIGH ALCH FUNC --------
def highAlch(cash: torch.Tensor) -> torch.Tensor:
    # chance to keep, 1=keep, 0=toss
    roll = torch.rand_like(cash) # 0.0 - <1.0
    #print(f"roll: {roll}")

    # logic, keep=+ALCH_VALUE, lose=ALCH_VALUE-ITEM_COST
    change = torch.where(roll <= .65, ALCH_VALUE, ALCH_VALUE - ITEM_COST)
    
    # check gamblers ruin (if it has 1 item), 1=continue, 0=gamblers ruin
    keepGoing = torch.where(cash >= ITEM_COST, 1, 0)
    #print(f"go: {keepGoing}")
    
    # mult change by whether can keep going
    change.mul_(keepGoing)
    #print(f"cva: {change}")
    
    # add change
    cash.add_(change)
    #print(f"new cash: {cash}\n")
    
    return cash
    

# -------- MAIN --------
# cash starting
xAxis = torch.arange( # range 1-START_COUNT (incl.)
    start=1, 
    end=START_COUNT + 1,
    device=devc, 
    dtype=torch.float32
)
cash = xAxis.view([1, -1]).repeat(repeats=[NUM_SIMS, 1]) # out=[NUM_SIMS, START_COUNT]
cash.mul_(ITEM_COST) # multiply starting value (total start value)
#print(f"{cash}")

# sequence thru alchs
for _ in range(ITER_ALCHS):
    cash = highAlch(cash) # more like cast = highAlch amirite??


# calc percentage of gambler ruined
losers = torch.where(cash < ITEM_COST, 1.0, 0.0) # 0>-1, else>0, 0=lost
losers = torch.sum(losers, dim=0) # sum
losers.div_(NUM_SIMS) # div / total
losers.mul_(100)
print(f"losers:{losers}")

# graph
gp = MultiLineGraph(
    x_axis_data=xAxis.cpu(), 
    y_axis_data_arr=[losers.cpu()], 
    legend=[
        ""
    ],
    x_label="number of starting items",
    y_label="% chance to fall into gamblers ruin",
    graph_title="Probability of becoming gambler-ruined based on GOLD LEAF"
)
gp.freeze_window()