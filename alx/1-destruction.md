(tinyllm) ray@i7-13700F tiny $ python scripts/destruction_experiments.py
Device:     cuda
Checkpoint: runs/austen-byte/checkpoints/best.pt
Iters:      200  Seed: 42

========================================================================
BASELINE
========================================================================
  Book                        Val Loss  Perplexity
  --------------------------------------------------
  Pride and Prejudice           1.4305        4.18
  Sense and Sensibility         1.4724        4.36
  Emma                          1.4220        4.15
  Mansfield Park                1.4774        4.38
  Persuasion                    1.4312        4.18
  Northanger Abbey              1.4811        4.40
  Sherlock Holmes               1.6764        5.35
  Alice in Wonderland           1.7941        6.01
  --------------------------------------------------
  Average                       1.5231

========================================================================
CONTEXT WINDOW PROBE
========================================================================
  Context     Avg Loss    Delta   Delta%   Marginal
  -------------------------------------------------------
  context_1       3.2662  +1.7431  +114.4%          ▒
  context_2       2.9245  +1.4013   +92.0%    -0.3417
  context_4       2.3916  +0.8685   +57.0%    -0.5329
  context_8       2.0018  +0.4787   +31.4%    -0.3898
  context_16      1.8578  +0.3347   +22.0%    -0.1440
  context_32      1.7994  +0.2762   +18.1%    -0.0585
  context_64      1.6734  +0.1502    +9.9%    -0.1260
  context_128     1.5231  +0.0000    +0.0%    -0.1502

========================================================================
DESTRUCTION EXPERIMENTS
========================================================================

  Experiment: shuffle_letters
  Book                        Baseline  Corrupted    Delta   Delta%
  --------------------------------------------------------------------
!! Pride and Prejudice           1.4305     4.1729  +2.7424  +191.7%
!! Sense and Sensibility         1.4724     4.1982  +2.7258  +185.1%
!! Emma                          1.4220     4.1845  +2.7625  +194.3%
!! Mansfield Park                1.4774     4.2506  +2.7733  +187.7%
!! Persuasion                    1.4312     4.2096  +2.7784  +194.1%
!! Northanger Abbey              1.4811     4.2016  +2.7205  +183.7%
!! Sherlock Holmes               1.6764     4.1490  +2.4727  +147.5%
!! Alice in Wonderland           1.7941     4.0382  +2.2440  +125.1%
  --------------------------------------------------------------------
   Average                       1.5231     4.1756  +2.6524  +174.1%

  Experiment: shuffle_middle
  Book                        Baseline  Corrupted    Delta   Delta%
  --------------------------------------------------------------------
!! Pride and Prejudice           1.4305     2.8450  +1.4146   +98.9%
!! Sense and Sensibility         1.4724     2.9167  +1.4443   +98.1%
!! Emma                          1.4220     2.8434  +1.4214  +100.0%
!! Mansfield Park                1.4774     2.9205  +1.4431   +97.7%
!! Persuasion                    1.4312     2.8271  +1.3959   +97.5%
!! Northanger Abbey              1.4811     2.8943  +1.4132   +95.4%
!! Sherlock Holmes               1.6764     2.8646  +1.1883   +70.9%
!! Alice in Wonderland           1.7941     2.8101  +1.0160   +56.6%
  --------------------------------------------------------------------
   Average                       1.5231     2.8652  +1.3421   +88.1%

  Experiment: shuffle_words
  Book                        Baseline  Corrupted    Delta   Delta%
  --------------------------------------------------------------------
 ! Pride and Prejudice           1.4305     1.7717  +0.3412   +23.9%
 ! Sense and Sensibility         1.4724     1.7359  +0.2634   +17.9%
 ! Emma                          1.4220     1.7109  +0.2889   +20.3%
 ! Mansfield Park                1.4774     1.7590  +0.2817   +19.1%
 ! Persuasion                    1.4312     1.7474  +0.3162   +22.1%
 ! Northanger Abbey              1.4811     1.7933  +0.3122   +21.1%
 ! Sherlock Holmes               1.6764     1.9566  +0.2803   +16.7%
 ! Alice in Wonderland           1.7941     2.0651  +0.2710   +15.1%
  --------------------------------------------------------------------
   Average                       1.5231     1.8175  +0.2944   +19.3%

  Experiment: reverse
  Book                        Baseline  Corrupted    Delta   Delta%
  --------------------------------------------------------------------
!! Pride and Prejudice           1.4305     4.9480  +3.5175  +245.9%
!! Sense and Sensibility         1.4724     4.9722  +3.4998  +237.7%
!! Emma                          1.4220     5.1009  +3.6789  +258.7%
!! Mansfield Park                1.4774     4.8581  +3.3807  +228.8%
!! Persuasion                    1.4312     5.0038  +3.5726  +249.6%
!! Northanger Abbey              1.4811     4.9572  +3.4761  +234.7%
!! Sherlock Holmes               1.6764     5.1068  +3.4304  +204.6%
!! Alice in Wonderland           1.7941     5.7142  +3.9201  +218.5%
  --------------------------------------------------------------------
   Average                       1.5231     5.0826  +3.5595  +233.7%

  Experiment: random_letters
  Book                        Baseline  Corrupted    Delta   Delta%
  --------------------------------------------------------------------
!! Pride and Prejudice           1.4305     6.1946  +4.7641  +333.0%
!! Sense and Sensibility         1.4724     6.2670  +4.7945  +325.6%
!! Emma                          1.4220     6.2130  +4.7910  +336.9%
!! Mansfield Park                1.4774     6.3430  +4.8657  +329.4%
!! Persuasion                    1.4312     6.2487  +4.8175  +336.6%
!! Northanger Abbey              1.4811     6.2343  +4.7532  +320.9%
!! Sherlock Holmes               1.6764     6.1508  +4.4744  +266.9%
!! Alice in Wonderland           1.7941     5.8298  +4.0357  +224.9%
  --------------------------------------------------------------------
   Average                       1.5231     6.1851  +4.6620  +306.1%

  Experiment: replace_names
  Book                        Baseline  Corrupted    Delta   Delta%
  --------------------------------------------------------------------
   Pride and Prejudice           1.4305     1.5174  +0.0869    +6.1%
   Sense and Sensibility         1.4724     1.5417  +0.0693    +4.7%
   Emma                          1.4220     1.5117  +0.0897    +6.3%
   Mansfield Park                1.4774     1.5308  +0.0535    +3.6%
   Persuasion                    1.4312     1.4917  +0.0605    +4.2%
   Northanger Abbey              1.4811     1.5562  +0.0751    +5.1%
   Sherlock Holmes               1.6764     1.6764  +0.0000    +0.0%
   Alice in Wonderland           1.7941     1.7941  +0.0000    +0.0%
  --------------------------------------------------------------------
   Average                       1.5231     1.5775  +0.0544    +3.6%
(tinyllm) ray@i7-13700F tiny $