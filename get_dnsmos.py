import soundfile as sf
import sys
import os
from metrics.dnnmos_metric import compute_dnsmos

 
for f in os.listdir(sys.argv[1]):
    in_audio,fs = sf.read(os.path.join(sys.argv[1],f))


    dnsmos_score = compute_dnsmos(in_audio,fs)
    print(dnsmos_score)