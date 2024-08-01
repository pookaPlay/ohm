from bls.ADD import ADD

def NOT(x):
    return 1 - x

def get_window_indices(base_index, window_width, list_length):
    """
    Returns a list of indices into a list, centered at the base_index with a specified window width.
    Indices wrap around if they go out of bounds.
    
    Parameters:
    - base_index (int): The central index of the window.
    - window_width (int): The width of the window.
    - list_length (int): The length of the list.
    
    Returns:
    - List[int]: A list of indices within the specified window.
    """
    radius = window_width // 2
    if window_width % 2 == 0:
        start_index = base_index - radius
        end_index = base_index + radius
    else:
        start_index = base_index - radius
        end_index = base_index + radius + 1

    indices = []
    for i in range(start_index, end_index):
        wrapped_index = i % list_length
        indices.append(wrapped_index)
    return indices

def get_reflected_indices(base_index, window_width, list_length):
    """
    Returns a list of indices into a list, centered at the base_index with a specified window width.
    Indices are reflected if they go out of bounds.
    
    Parameters:
    - base_index (int): The central index of the window.
    - window_width (int): The width of the window.
    - list_length (int): The length of the list.
    
    Returns:
    - List[int]: A list of indices within the specified window.
    """
    radius = window_width // 2
    start_index = base_index - radius
    end_index = base_index + radius + 1

    indices = []
    for i in range(start_index, end_index):
        if i < 0:
            reflected_index = -i - 1
        elif i >= list_length:
            reflected_index = 2 * list_length - i - 1
        else:
            reflected_index = i
        indices.append(reflected_index)
    return indices

class OHM_ADDER_CHANNEL:

    def __init__(self,  numInputsMirrored, memD, adderInstance) -> None:        
        # numInputsMirrored is 2*actual (+ve/-ve)
        # adderInstance is used to setup connections
        self.numInputsMirrored = numInputsMirrored
        self.numInputs = int(self.numInputsMirrored/2)        
        self.numOutputs = self.numInputsMirrored
        self.memD = memD
        
        self.adders = [ADD() for _ in range(self.numInputsMirrored)]        
        
        # connect and mirror inputs
        
        # Connect first so many 
        #self.inIndexA = list(range(self.numInputs))        
        
        # 1D convolution with wrapping
        # self.inIndexA = get_window_indices(adderInstance, self.numInputs, self.memD)                
        self.inIndexA = get_reflected_indices(adderInstance, self.numInputs, self.memD)                
        self.inIndexA.extend(self.inIndexA)
        
        # parameter memory (B) should be 2*A 
        self.inIndexB = list(range(self.numInputsMirrored))
        self.outIndex = list(range(self.numOutputs))
            

    def Reset(self) -> None:        
        for ai in range(len(self.adders)):
            self.adders[ai].Reset()
                        
    def Output(self):        
        return self.denseOut

    def Calc(self, memA, memB, lsb=0) -> None:
    
        self.aInputs = [memA.Output(aIndex) for aIndex in self.inIndexA]
        # mirror inputs
        for ai in range(self.numInputs):
            self.aInputs[ai+self.numInputs] = NOT(self.aInputs[ai])

        self.bInputs = [memB.Output(bIndex) for bIndex in self.inIndexB]

        for ai in range(len(self.adders)):
            self.adders[ai].Calc(self.aInputs[ai], self.bInputs[ai], lsb)
        
        self.denseOut = [ad.Output() for ad in self.adders]        

    def Step(self) -> None:
        for ai in range(len(self.adders)):
            self.adders[ai].Step()
        
    def Print(self, prefix="", verbose=1) -> None:        
        print(f"{prefix}OHM_ADDER_CHANNEL: {len(self.adders)} adders")
        for ai in range(len(self.adders)):
            self.adders[ai].Print(prefix + "  ", verbose)

