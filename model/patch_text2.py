b = open('build_manuscript.py', encoding='utf-8').read()
def rep(old, new):
    global b
    assert old in b, old[:60]
    b = b.replace(old, new)

# derived cross-seed quantities used in text
rep("tau = lambda *p: RX.ac(*p)\n", """tau = lambda *p: RX.ac(*p)
gap = tau('perm_null', 'rho_obs') - tau('perm_null', 'rho_null_mean'); n_null_ok = int((tau('perm_null', 'p_greater') >= 0.05).sum())
rho_e_ita = np.array([XA[s]['skin']['v1_all']['spearman_ita_eshuf']['rho'] for s in XA]); kwe_ita_sig = int(sum(XA[s]['skin']['v1_all']['kw_e_shuf']['p'] < 0.05 for s in XA))
rho_lsas_ig = tau('faithfulness', 'mean', 'rho_lsas_IG', 'rho'); rho_lsas_dg = tau('faithfulness', 'mean', 'rho_lsas_DG', 'rho'); rho_e_ig = tau('faithfulness', 'mean', 'rho_eshuf_IG', 'rho')
frac_dg = tau('faithfulness', 'mean', 'frac_DG_pos'); frac_ig = tau('faithfulness', 'mean', 'frac_IG_pos')
""")
rep("""is no larger than for randomly re-paired saliency maps (null ρ = {pn['rho_null_mean']:.2f}); the area-adjusted excess is small (median {R['enrichment_vs_null']['all']['median_e_shuf']:+.3f}) and not size-monotonic.\"""",
    """is indistinguishable from that of randomly re-paired saliency maps (null ρ = {pn['rho_null_mean']:.2f}) in {n_null_ok}/{K} seeds and exceeds it by ≤ {gap.max():.3f} in the rest; the area-adjusted excess is small (median {R['enrichment_vs_null']['all']['median_e_shuf']:+.3f}) and decreases with lesion size.\"""")
rep("""f"Localization is not faithfulness: raw LSAS did not predict deletion or insertion gains, area-adjusted localization did so only moderately (ρ ≈ {fm['rho_eshuf_IG']['rho']:.2f} with the insertion gain), and deletion evidence was weak and baseline-dependent.\"""",
    """f"Localization is not faithfulness: raw LSAS was unrelated or negatively related to the deletion and insertion gains (ρ with insertion gain {rho_lsas_ig.min():+.2f} to {rho_lsas_ig.max():+.2f}; with deletion gain {rho_lsas_dg.min():+.2f} to {rho_lsas_dg.max():+.2f} across seeds), area-adjusted localization tracked insertion faithfulness moderately (ρ = {rho_e_ig.min():.2f}–{rho_e_ig.max():.2f}), and deletion evidence was weak and baseline-dependent.\"""")
rep("""f"No association between a skin-tone proxy and localization was detected, but the darkest stratum has only {R['skin']['v1_correct_only']['n']['V-VI']} correctly classified images; this is insufficient evidence, not fairness.\"""",
    """f"Skin-tone-proxy findings are inconclusive: the darkest stratum holds only {R['skin']['v1_correct_only']['n']['V-VI']} correctly classified images, the raw difference follows lesion geometry, and area-adjusted associations change sign between seeds (ρ = {rho_e_ita.min():+.2f} to {rho_e_ita.max():+.2f}); this is insufficient evidence, not fairness.\"""")
rep("""but this association was no stronger than under randomly paired maps (permutation p = {pn['p_greater']:.2f}); after subtracting""",
    """but this association was indistinguishable from that under randomly paired maps in {n_null_ok} of {K} seeds (exceeding it by at most {gap.max():.3f} in Spearman ρ in the remaining seed); after subtracting""")
rep("""and not monotonic in size. LSAS did not differ""", """and decreased with lesion size in every seed. LSAS did not differ""")
rep("""raw LSAS did not predict faithfulness. Median saliency stability was {R['sts']['median']:.2f}. No skin-tone-proxy association was detected, but only {R['skin']['v1_correct_only']['n']['V-VI']} darkest-stratum images were correctly classified and the minimum detectable effect was large (d = {R['skin']['mde_correct_only']['cohen_d_80pct']:.2f}).""",
    """raw LSAS did not predict faithfulness. Median saliency stability was {RX.rng_s(tau('sts', 'median'), 2)} across seeds. Skin-tone-proxy results were inconclusive: the darkest stratum held only {R['skin']['v1_correct_only']['n']['V-VI']} correctly classified images, the minimum detectable effect was large (d = {R['skin']['mde_correct_only']['cohen_d_80pct']:.2f}), raw differences followed lesion geometry, and area-adjusted associations were small, significant in {kwe_ita_sig} of {K} seeds for stratum differences, and changed sign between seeds.""")
rep("and absent evidence of a skin-tone effect is not fairness.", "and inconclusive subgroup results are not evidence of fairness.")
rep("""The analysis found no association between ITA-proxy tone and localization, but this is neither positive nor reassuring evidence. The darkest stratum""",
    """The analysis found no consistent association between ITA-proxy tone and area-adjusted localization: the raw difference follows lesion geometry, area-adjusted correlations ranged from ρ = {rho_e_ita.min():+.2f} to {rho_e_ita.max():+.2f} across seeds with both signs, and stratum differences reached nominal significance in {kwe_ita_sig} of {K} seeds. This is neither positive nor reassuring evidence, and the seed-to-seed instability is a warning against single-run subgroup conclusions. The darkest stratum""")
rep("""M.p("The analysis found no consistent""", """M.p(f"The analysis found no consistent""")
rep("A defensible practice", "A defensible practice") if False else None
rep("""In {cnt""", """In {cnt""") if False else None
open('build_manuscript.py', 'w', encoding='utf-8').write(b)
print('ok')
