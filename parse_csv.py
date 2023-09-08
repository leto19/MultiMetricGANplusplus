import pandas as pd
import sys


in_df = pd.read_csv(sys.argv[1])


if "SIG_MOS" in in_df:
    avg_dns_mos_sig = in_df["SIG_MOS"].mean()

    avg_dns_mos_bak = in_df["BAK_MOS"].mean()

    avg_dns_mos_ovr = in_df["OVR_MOS"].mean()

    print("AVERAGE SIG:",avg_dns_mos_sig)
    print("AVERAGE BAK:",avg_dns_mos_bak)
    print("AVERAGE OVR:",avg_dns_mos_ovr)
else:
    avg_sisdr = in_df["SI-SDR"].mean()
    print("AVERAGE SI-SDR",avg_sisdr)
    