"""DeBruijnGenerator.py: Defines a simple class for generating
deBruijn sequences one symbol at a time.
It is based on a class I originally wrote some years ago in c++.
A function that generates the entire sequence is also provided
for testing purposes.
Developed and tested using python v2.7.10 and numpy v1.15.4
Brendan Harding, March 2019
"""

__author__    = "Brendan Harding"
__copyright__ = "Copyright 2019"
__license__   = "MIT"
__version__   = "0.1.0"

import numpy as np

class DeBruijnGenerator:
    """
    Class for producing symbols from a debruijn sequence one at a time.
    This has the advantage of not generating the whole sequence in memory.
    The sequence produced is the lexicographic smallest.
    Functions are included to enable a checkpoint/restart like functionality.
    """
    
    def __init__(self,b,n):
        """Initialise the sequence given a base b and sub-sequence length n."""
        self.b = b                        # base
        self.n = n                        # length of subsequences
        self.i = n                        # internal working index
        self.m = 1                        # internal cache index
        self.a = np.zeros(n+1,np.int)     # working array
        self.cache = np.zeros(n+1,np.int) # internal cache
        self.completed = False            # a completion flag
        
    def reset(self):
        """Reset the sequence to start from the beginning."""
        self.a *= 0 
        self.cache *= 0
        self.i = self.n 
        self.m = 1 
        
    def __call__(self):
        """Return the next element in the sequence."""
        if self.m>0:
            self.m -= 1
            return self.cache[self.m+1]
        else:
            while self.i>0:
                self.a[self.i] += 1
                for j in range(1,self.n+1-self.i):
                    self.a[j+self.i] = self.a[j]
                #self.a[self.i+1:] = self.a[1:self.n+1-self.i] 
                # Should not use slicing here ^^^ (sequential only)
                if self.n%self.i==0:
                    self.cache[self.i:0:-1] = self.a[1:self.i+1] 
                    self.m = self.i-1
                    self.i = self.n-np.argmin(self.a[::-1]==self.b-1)
                    return self.cache[self.m+1]
                self.i = self.n-np.argmin(self.a[::-1]==self.b-1)
            self.completed = True
            self.reset()
            self.m -= 1
            return self.cache[self.m+1]
            
    def is_complete(self):
        """Return a flag specifying if sequence has run to completion."""
        return self.completed
        
    def reset_completion_flag(self): 
        """Resets the completion flag for the sequence."""
        self.completed = False
        
    def length(self):
        """Returns the sequence length, but beware of overflow!"""
        return self.b**self.n
        
    def dump_state(self):
        """Dump the current state of the generator (i.e. a checkpoint)."""
        return [self.b,self.n,self.i,self.m,self.a,self.cache,self.completed]
        
    def set_state(self,data):
        """Set the state of the generator (i.e. a restart from checkpoint)."""
        self.b = data[0]
        self.n = data[1]
        self.i = data[2]
        self.m = data[3]
        self.a = data[4]
        self.cache = data[5]
        self.completed = data[6]
        # There should probably be some checkes that self.a 
        # and self.cache are the correct length...
        
    # end class

# We don't really use this, other than to test the above.
def deBruijn(k, n):
    """
    Generate a de Bruijn sequence for alphabet k and subsequences of length n.
    This is from the Wikipedia article (based on that in Frank Ruskey's book).
    """
    try:
        _ = int(k) # check if k can be cast to an integer
        alphabet = list(map(str, range(k))) # if so, make our alphabet a list
    except (ValueError, TypeError):
        alphabet = k
        k = len(k)
    a = [0] * k * n
    sequence = []
    def db(t, p):
        if t > n:
            if n % p == 0:
                sequence.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)
    db(1, 1)
    return "".join(alphabet[i] for i in sequence)

if __name__ == "__main__":
	# If ran as a script, perform a few short tests
	db_2_12 = DeBruijnGenerator(2,12)
	seq1 = np.array([db_2_12() for _ in range(2**12)])
	seq2 = np.array([int(symbol) for symbol in deBruijn(2,12)])
	assert (seq1==seq2).all()
	db_3_8 = DeBruijnGenerator(3,8)
	seq1 = np.array([db_3_8() for _ in range(3**8)])
	seq2 = np.array([int(symbol) for symbol in deBruijn(3,8)])
	assert (seq1==seq2).all()
	db_4_6 = DeBruijnGenerator(4,6)
	seq1 = np.array([db_4_6() for _ in range(4**6)])
	seq2 = np.array([int(symbol) for symbol in deBruijn(4,6)])
	assert (seq1==seq2).all()
	db_5_5 = DeBruijnGenerator(5,5)
	seq1 = np.array([db_5_5() for _ in range(5**5)])
	seq2 = np.array([int(symbol) for symbol in deBruijn(5,5)])
	assert (seq1==seq2).all()
	print("DeBruijnGenerator.py: all tests passed!")
