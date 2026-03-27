MFCC and dtw from scratch (with minimal library functions)  
Following these steps:  
1. pre emphasis
2. frame blocking  
3. hamming windows  
4. FFT  
5. mel filter  
6. DCT  
7. add delta component  
8. dtw compute best cost
  
~~To do: port this to FPGA platform, hence the need to implement it from scratch.~~  
DTW has poor generalization ability based on empirical results.  
Thus I've turned to HMM model and Viterbi algorithm.  
HMM is in the range of machine learning, which is a data based model, can be generalizable with some more data.  
Still MFCC is needed as a way of feature extraction.  
HMM is here. 
