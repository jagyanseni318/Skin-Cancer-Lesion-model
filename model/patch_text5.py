def patch(fn, pairs):
    s = open(fn, encoding='utf-8').read()
    for a, b in pairs:
        assert a in s, (fn, a[:80])
        s = s.replace(a, b)
    open(fn, 'w', encoding='utf-8').write(s)


patch('ms_results_xai.py', [
    ("    v1c = R['skin']['v1_correct_only']; v1a = R['skin']['v1_all']; mde = R['skin']['mde_correct_only']\n",
     "    v1c = R['skin']['v1_correct_only']; v1a = R['skin']['v1_all']; mde = R['skin']['mde_correct_only']\n"
     "    dref = pd.read_csv(RES / 'ref_per_image_enriched.csv'); ga = dref[dref.fst_v1 == 'I-II']; gb = dref[dref.fst_v1 == 'V-VI']\n"
     "    from scipy import stats as _st\n"
     "    p_area = _st.mannwhitneyu(ga.area224, gb.area224).pvalue\n"),
    ("but this direction is that of the area geometry, since darker-proxy images tend to have smaller lesions (ρ(ITA, area) = {v1a['spearman_ita_area']['rho']:+.3f} in the reference seed).",
     "but this direction follows lesion geometry: in the reference seed the darkest-stratum lesions were smaller (median area fraction {gb.area224.median():.3f} vs {ga.area224.median():.3f} for I–II; Mann–Whitney {fp(p_area)}), and the shuffled-saliency null differed by a similar amount (median {gb.null_shuf_mean.median():.3f} vs {ga.null_shuf_mean.median():.3f}), whereas the continuous ITA–area correlation is near zero (ρ = {v1a['spearman_ita_area']['rho']:+.3f}) because the stratum difference is confined to the small darkest group."),
    ("which shows that the proxy itself is unstable for the darkest stratum: many v1 'V–VI' images are not distinguishable from lesion-adjacent shading or vignette effects.",
     "which shows that the proxy itself is sensitive to the definition, most strongly for the darkest stratum (377 vs 237 images); the proxy values of these images may partly reflect illumination, shading or vignette, but this was not verified against any independent skin-type label."),
])
patch('build_manuscript.py', [
    ("and what remains after subtraction is a small, non-monotonic excess.", "and what remains after subtraction is a small excess that decreases, rather than increases, with lesion size."),
    ("a five-fold larger Fréchet distance", "an approximately four-fold larger Fréchet distance"),
    ("and stratum differences reached nominal significance in {kwe_ita_sig} of {K} seeds. This is neither",
     "stratum differences reached nominal significance in {kwe_ita_sig} of {K} seeds, and the correlation of ITA with excess LSAS reached nominal significance in {int(sum(XA[s]['skin']['v1_all']['spearman_ita_eshuf']['p'] < 0.05 for s in XA))} of {K} seeds (with both signs). This is neither"),
    ("The proxy analysis should be read as a check that a large tone-dependent artefact is not present, not as fairness validation",
     "The proxy analysis should be read as a check that no large, consistent tone-dependent artefact was found, not as fairness validation"),
    ("Localization in an area-insensitive form was modestly worse for wrong predictions, and insertion faithfulness was lower, consistent with wrong predictions resting on less coherent evidence, but these effects are small and cannot serve as error detectors.",
     "Localization in an area-insensitive form was modestly worse for wrong predictions in some seeds (significant in {int((tau('correct_vs_incorrect', 'loc_auc', 'p') < 0.05).sum())} of {K}), and insertion faithfulness was lower in {int((tau('faithfulness', 'mean', 'IG_corr_vs_incorr', 'p') < 0.05).sum())} of {K} seeds, consistent with wrong predictions resting on less coherent evidence, but these effects are small and cannot serve as error detectors."),
    ("M.p(\"Including misclassified predictions", "M.p(f\"Including misclassified predictions"),
])
print('ok')
