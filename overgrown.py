import torch
from eco_5.graph import MultiLineGraph
devc = torch.device(type="cuda")
torch.set_printoptions(sci_mode=False)


# -------- CONSTS & SETUP --------
NUM_SIMS = 16000

# start with 20 ranarrs
seedPouch = torch.zeros([NUM_SIMS], device=devc, dtype=torch.float32)
seedPouch.add_(20)

# track, -1=did not run out of ranarrs yet, else=iteration it ran out on
tracked = torch.zeros([NUM_SIMS], device=devc, dtype=torch.float32)
tracked.add_(-1)


# -------- LOOP --------
# for each iteration, has 75% chance to keep
for i in range(1, 180): # start on 1 since if that 1st loop took seed it will have had 1 planting
    chanceSub = torch.rand([NUM_SIMS], device=devc, dtype=torch.float32)
    #print(f"{r}")
    chanceSub = torch.where(chanceSub <= .75, 0, -1)
    #print(f"{a}")
    #print(f"{r}")
    
    # sub chance to remove seed
    seedPouch.add_(chanceSub)
    #print(f"{a}")
    
    # track it by seeing 0s
    tracked = torch.where(seedPouch == 0, i, tracked)
    # set seedPouch -1 so it doesn't keep injecting i
    seedPouch = torch.where(seedPouch == 0, -1, seedPouch)

#print(f"{t}")


# -------- STATS --------
# ave
ave = torch.sum(tracked, dim=0)
ave.div_(tracked.size()[0])
print(f"ave: {ave}")

# worst
print(f"worst: {torch.min(tracked)}")

# best
print(f"best: {torch.max(tracked)}")

# median
med = torch.quantile(tracked, .5, interpolation="nearest")
print(f"median: {med}")