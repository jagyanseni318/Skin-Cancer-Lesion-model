def patch(fn, pairs):
    s = open(fn, encoding='utf-8').read()
    for a, b in pairs:
        assert a in s, (fn, a[:70])
        s = s.replace(a, b)
    open(fn, 'w', encoding='utf-8').write(s)


patch('ms_results_cls.py', [
    ("No seed was excluded, and the first-listed seed is not the best-performing one by construction (Table 2).",
     "No seed was excluded or selected. The reference seed (42), designated before the runs for the detailed per-image tables, is in fact the best of the five on accuracy ({df.acc.max():.3f}) and macro-F1 ({df.f1.max():.3f}) and is above the seed mean on most metrics (Table 2); its detailed results are therefore slightly optimistic relative to the seed mean, which is why every headline statistic is also reported for all seeds."),
    ("temperature scaling therefore {'did not improve' if agg['ece']['mean'] >= agg['ece_uncal']['mean'] else 'improved'} top-label ECE on the test set in {worse if agg['ece']['mean'] >= agg['ece_uncal']['mean'] else k - worse} of {k} seeds' comparisons ({'ECE increased' if worse > k / 2 else 'ECE decreased'} in {worse}/{k} seeds), and NLL",
     "temperature scaling therefore {'did not improve' if agg['ece']['mean'] >= agg['ece_uncal']['mean'] else 'improved'} top-label ECE on the test set (ECE was higher after scaling in {worse}/{k} seeds), and NLL"),
    ("The likely explanation is that the focal + label-smoothing objective already yields modest confidence and that a 503-image calibration set with 7–8 images in the smallest classes gives a noisy single-parameter fit.",
     "Two plausible explanations, which were not tested, are that the focal plus label-smoothing objective already yields modest confidence and that a 503-image calibration set with 7–8 images in the smallest classes gives a noisy single-parameter fit."),
])
patch('ms_results_xai.py', [
    ("The class ordering is not stable enough across seeds to be interpreted beyond 'classes differ': the Kruskal–Wallis test on excess LSAS was significant at α = 0.05 in {cnt(ac('kw_class_all_e_shuf', 'p'), lambda v: v < 0.05)}/{K} seeds.",
     "The Kruskal–Wallis test on excess LSAS was significant at α = 0.05 in {cnt(ac('kw_class_all_e_shuf', 'p'), lambda v: v < 0.05)}/{K} seeds, so class differences are reproducible as a statement that classes differ; the confidence intervals of the rarer classes are too wide to interpret the ordering of individual classes."),
])
print('ok')
