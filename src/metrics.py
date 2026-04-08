from pymcd.mcd import Calculate_MCD
from pystoi import stoi

def MCD(orig_audio, rec_audio, sr):
    # three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics
    mcd_toolbox = Calculate_MCD(MCD_mode="plain")
    mcd = mcd_toolbox.calculate_mcd(orig_audio, rec_audio)
    
def STOI(orig_audio, rec_audio, sr):
    stoi = stoi(orig_audio, rec_audio, sr, extended=False)

def eSTOI(orig_audio, rec_audio, sr):
    stoi = stoi(orig_audio, rec_audio, sr, extended=True)
