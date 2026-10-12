"""Assemble the final manuscript (DOCX + Markdown) from result files. Run after all experiments finished."""
import json, sys
import numpy as np, pandas as pd
from common import *
import ms_static, ms_results_cls as RC, ms_results_xai as RX, ms_response as RP, ms_models_int8 as MD, ms_gui, ms_adapt
from ms_lib import Manuscript
from ms_results_cls import pm, fp

RES = OUT / 'results'; FIG = OUT / 'figures'
M = Manuscript(); WARN = []


def check(cond, msg):
    if not cond: WARN.append(msg); print('CLAIM-CHECK FAILED:', msg)


S = json.load(open(RES / 'cls_summary.json')); agg = S['agg']; K = len(S['seeds']); XA = RX.XA; REF = 42; R = XA[REF]; PA = RX.PA
assert ms_adapt.available(), 'run pad_finetune.py first'
AD = ms_adapt.numbers()
sm = R['size_measures']; pn = R['perm_null']; fm = R['faithfulness']['mean']
cv = pd.read_csv(RES / 'convergence.csv')
nfs = json.load(open(RES / 'null_faith_summary.json'))
# ---------------------------------------------------------- claim checks (narrative depends on these holding across seeds)
check(sum(XA[s]['perm_null']['p_greater'] < 0.05 for s in XA) <= 1 and max(XA[s]['perm_null']['rho_obs'] - XA[s]['perm_null']['rho_null_mean'] for s in XA) < 0.02, 'raw LSAS-area correlation exceeds geometric null materially')
check(all(XA[s]['size_measures']['e_shuf']['spearman_area']['rho'] < 0 for s in XA), 'excess-LSAS vs area not negative in all seeds')
check(all(XA[s]['correct_vs_incorrect']['lsas']['p'] > 0.05 for s in XA), 'LSAS differs by correctness in some seed')
check(all(XA[s]['faithfulness']['mean']['wilcoxon_IG_p'] < 0.05 for s in XA), 'insertion gain not significant in some seed')
check(all(XA[s]['sts']['frac_below_05'] < 0.1 for s in XA), 'many unstable maps in some seed')
check(nfs['EIG']['median_range'][0] > 0 and nfs['EDG']['median_range'][0] > 0, 'median EIG or EDG not positive in every seed')
check(nfs['EIG']['n_seeds_wilcoxon_sig'] == nfs['EIG']['n_seeds'] and nfs['EDG']['n_seeds_wilcoxon_sig'] == nfs['EDG']['n_seeds'], 'EIG or EDG not Wilcoxon-significant in every seed')
tau = lambda *p: RX.ac(*p)
gap = tau('perm_null', 'rho_obs') - tau('perm_null', 'rho_null_mean'); n_null_ok = int((tau('perm_null', 'p_greater') >= 0.05).sum())
rho_e_ita = np.array([XA[s]['skin']['v1_all']['spearman_ita_eshuf']['rho'] for s in XA]); kwe_ita_sig = int(sum(XA[s]['skin']['v1_all']['kw_e_shuf']['p'] < 0.05 for s in XA))
rho_lsas_ig = tau('faithfulness', 'mean', 'rho_lsas_IG', 'rho'); rho_lsas_dg = tau('faithfulness', 'mean', 'rho_lsas_DG', 'rho'); rho_e_ig = tau('faithfulness', 'mean', 'rho_eshuf_IG', 'rho')
frac_dg = tau('faithfulness', 'mean', 'frac_DG_pos'); frac_ig = tau('faithfulness', 'mean', 'frac_IG_pos')


# ---------------------------------------------------------- quantities for the three-configuration comparison (Sections 4.12-4.15)
CF = MD.CFG; dn = MD.exd('densenet121'); rn = MD.exd('resnet50'); n_dn, n_rn = len(dn), len(rn); S3 = MD.compute_summary(); EFFJ = MD.EFF
mk = lambda k, m, d=3: f"{CF[k][m]['mean']:.{d}f} (SD {CF[k][m]['sd']:.{d}f})"
cnt_sig = lambda df, col, p, sign=1: int(((df[col] * sign > 0) & (df[p] < 0.05)).sum())
rn_eig_sig, rn_edg_sig = cnt_sig(rn, 'EIG_median', 'EIG_wilcoxon_p'), cnt_sig(rn, 'EDG_median', 'EDG_wilcoxon_p')
dn_eig_sig, dn_edg_sig = cnt_sig(dn, 'EIG_median', 'EIG_wilcoxon_p'), cnt_sig(dn, 'EDG_median', 'EDG_wilcoxon_p')
prec = lambda a, kind: MD.CMP['tests'][f"precision_{a}_{kind}"]
fkind = {'dn': 'int8', 'rn': 'int8'}
dacc = {k: -prec('densenet121' if k == 'dn' else 'resnet50', fkind[k])['acc']['mean_diff_int8_minus_fp32'] * 100 for k in ('dn', 'rn')}
dF1 = {k: -prec('densenet121' if k == 'dn' else 'resnet50', fkind[k])['f1']['mean_diff_int8_minus_fp32'] * 100 for k in ('dn', 'rn')}
shrink = {k: (EFFJ[f'{k}_fp32']['size_mb'] / EFFJ[MD.FIN[k]]['size_mb']) for k in ('dn', 'rn')} if EFFJ else {'dn': float('nan'), 'rn': float('nan')}
speedup = {k: (EFFJ[f'{k}_fp32']['cpu_b32_ms'] / EFFJ[MD.FIN[k]]['cpu_b32_ms']) for k in ('dn', 'rn')} if EFFJ else {'dn': float('nan'), 'rn': float('nan')}
check(EFFJ is not None, 'efficiency.json is missing (run bench_efficiency.py on an idle machine)')
# the static statements about ResNet50 in Sections 1.5, 3.17, 4.11-4.12, 5.9 and the study-scope bullets are guarded here
check(rn_eig_sig <= 1 and rn_edg_sig <= 1 and rn.EIG_median.mean() < 0.2 * dn.EIG_median.mean() and rn.EDG_median.mean() < 0.2 * dn.EDG_median.mean(), 'static text says EIG/EDG did not replicate on ResNet50 (not consistently positive, effect below 20% of DenseNet121), but this no longer holds')
check(dn_eig_sig == n_dn and dn_edg_sig == n_dn, 'static text says EIG/EDG are significantly positive in every DenseNet121 seed')
check(cnt_sig(rn, 'e_shuf_median', 'e_shuf_wilcoxon_p', -1) == n_rn, 'static text says ResNet50 excess LSAS is significantly negative in every seed')
check(bool((rn.rho_lsas_area > 0.5).all() and (rn.IG_frac_pos > 0.5).all()), 'static text says the geometric confound and majority-positive insertion gain replicate on ResNet50')
check(n_dn == 5 and n_rn == 5, 'both architectures must have five seeds')

# ---------------------------------------------------------- front matter
M.h(0, 'Explanation Reliability Beyond Diagnostic Accuracy in Dermoscopic Skin-Lesion Classification')
M.p("Soumya Das¹, Vipul Virsinghbhai Gamit²*, Tanisha³, Jagyanseni Paikaraya⁴, Bharat Tank⁵, Vijay Jagdish Upadhye⁶", 'center')
M.p("¹,²,³,⁴ Department of Computer Applications (MCA), Faculty of IT and Computer Science, Parul University, Vadodara, Gujarat, 391760, India;", 'center')
M.p("⁵ Department of Computer Science and Engineering, PIET, Faculty of Engineering and Technology, Parul University, Vadodara, Gujarat, 391760, India;", 'center')
M.p("⁶ Department of Microbiology, Parul Institute of Applied Sciences and Research and Development Cell, Parul University, Vadodara, Gujarat, 391760, India", 'center')
M.p("*Corresponding author: vipulbhai.gamit26053@paruluniversity.ac.in", 'center')
M.h(1, 'Highlights')
M.bullets([
    f"Three configurations (DenseNet121, ResNet50 and INT8 versions of both, all explained with Grad-CAM++) were compared on identical data under a decision rule fixed in advance; none is best on every criterion ([[T:three]], [[F:bars]]).",
    (f"INT8 quantization made the models {shrink['dn']:.1f}- and {shrink['rn']:.1f}-fold smaller and {speedup['dn']:.1f}- and {speedup['rn']:.1f}-fold faster on CPU, with accuracy {CF[MD.FIN['dn']]['acc']['mean']:.3f} and {CF[MD.FIN['rn']]['acc']['mean']:.3f} against {CF['dn_fp32']['acc']['mean']:.3f} and {CF['rn_fp32']['acc']['mean']:.3f} for the FP32 parents, but required the first convolution and the head in FP32 ([[T:int8]])." if shrink else "INT8 quantization is evaluated in Section 4.13."),
    f"The lesion-size effect on localization (LSAS) is geometric: raw LSAS matches randomly re-paired maps in {n_null_ok}/{K} DenseNet121 seeds and the area-adjusted excess is small (median {R['enrichment_vs_null']['all']['median_e_shuf']:+.3f}); localization is not faithfulness.",
    f"The null-controlled excess insertion/deletion gain (Section 4.11) was significantly positive in {dn_eig_sig} of {n_dn} DenseNet121 seeds but {rn_eig_sig} of {n_rn} ResNet50 seeds: established for DenseNet121 with Grad-CAM++, not generally (Section 4.12).",
    "Skin-tone-proxy findings are inconclusive and are not evidence of fairness.",
    f"On PAD-UFES-20, class mapping and prevalence explain little of the accuracy drop (0.82 → {RX.ms(RX.ac('C_HAM_test_5class_images_5way_output', 'acc', src=PA))} → {RX.ms(RX.ac('E_PAD_mapped_5way_output', 'acc', src=PA))}); zero-shot accuracy is below the {AD['maj']:.3f} of always predicting the most frequent class, and fine-tuning on 30% of the target patients raised it to {AD['ft_acc']:.3f} (SD {AD['ft_acc_sd']:.3f}; [[T:pad_ft]]), so target-domain evaluation and adaptation are needed.",
])
f1s = [PA[s]['E_PAD_mapped_5way_output']['f1_present'] for s in PA]
M.h(1, 'Abstract')
M.p("**Background:** Clinical trust in dermoscopic AI cannot be inferred from accuracy alone: probabilities may be miscalibrated, explanations unreliable, behaviour skin-tone dependent and performance fragile under domain shift; localization metrics are confounded by lesion size.")
M.p("**Methods:** A DenseNet121 (primary), a ResNet50 and INT8-quantized versions of both were trained or derived on lesion-level HAM10000 splits (8,020/1,005/990 images; five seeds each) and compared under a decision rule fixed in advance. Grad-CAM++ explanations were audited with null-controlled localization (LSAS), deletion/insertion plus a new null-controlled faithfulness metric, MC-dropout stability, an ITA skin-tone proxy, and an additive PAD-UFES-20 decomposition with an adaptation experiment.")
M.p(f"**Results:** Accuracy was {CF['dn_fp32']['acc']['mean']:.3f} (SD {CF['dn_fp32']['acc']['sd']:.3f}) for DenseNet121 and {CF['rn_fp32']['acc']['mean']:.3f} (SD {CF['rn_fp32']['acc']['sd']:.3f}) for ResNet50; no configuration was best on every criterion. " +
    (f"The INT8 versions reached accuracy {CF[MD.FIN['dn']]['acc']['mean']:.3f} (DenseNet121) and {CF[MD.FIN['rn']]['acc']['mean']:.3f} (ResNet50) and were {shrink['dn']:.1f}- and {shrink['rn']:.1f}-fold smaller and {speedup['dn']:.1f}- and {speedup['rn']:.1f}-fold faster on CPU. " if shrink else "") +
    f"Raw LSAS rose steeply with lesion size (ρ = {sm['lsas']['spearman_area']['rho']:.2f}) but was indistinguishable from randomly paired maps in {n_null_ok} of {K} DenseNet121 seeds; the area-adjusted excess was small (median {R['enrichment_vs_null']['all']['median_e_shuf']:+.3f}). Insertion faithfulness exceeded random ordering for both architectures, deletion faithfulness was not supported, and the new excess insertion/deletion gain was significantly positive in {dn_eig_sig} of {n_dn} DenseNet121 seeds but in only {rn_eig_sig} of {n_rn} ResNet50 seeds (median EIG {dn.EIG_median.mean():+.3f} against {rn.EIG_median.mean():+.3f}). Skin-tone-proxy associations were small and sign-unstable, hence inconclusive. On PAD-UFES-20, zero-shot accuracy fell from {RX.ms(RX.ac('C_HAM_test_5class_images_5way_output', 'acc', src=PA))} to {RX.ms(RX.ac('E_PAD_mapped_5way_output', 'acc', src=PA))}, below the {AD['maj']:.3f} of always predicting the most frequent class, mostly through appearance and label-semantics shift rather than class mapping or prevalence; fine-tuning dense block 4 and the head on 30% of the target patients restored accuracy to {AD['ft_acc']:.3f} (SD {AD['ft_acc_sd']:.3f}) and balanced accuracy to {AD['ft_bacc']:.3f} (SD {AD['ft_bacc_sd']:.3f}) on the remaining 70% (mean over {AD['n_splits']} patient-level splits), so target-domain evaluation and adaptation are needed.")
M.p("**Conclusions:** Accuracy, calibration, localization, faithfulness, stability, subgroup behaviour, external generalization and efficiency are distinct properties, and the best model differs by criterion. Mask-overlap scores need null control for lesion geometry, faithfulness needs a plausible-alternative null, INT8 deployment needs validation-selected configurations, and inconclusive subgroup results are not evidence of fairness. These are computational findings, not clinical claims.")
M.h(1, 'Keywords'); M.p("Dermoscopic imaging; skin-lesion classification; explainable artificial intelligence; Grad-CAM++; localization versus faithfulness; deletion and insertion; calibration; lesion-size confounding; skin-tone proxy; domain shift; HAM10000; PAD-UFES-20")
M.h(1, 'Abbreviations')
M.p("ACK, actinic keratosis (PAD-UFES-20 label mapped to AKIEC); AKIEC, actinic keratosis/intraepithelial carcinoma; AUROC, area under the ROC curve; BCa, bias-corrected and accelerated; BCC, basal cell carcinoma; BKL, benign keratosis-like lesion; DF, dermatofibroma; DG, deletion gain; ECE, expected calibration error; EDG, excess deletion gain; EIG, excess insertion gain; FP32, 32-bit floating point; IG, insertion gain; INT8, 8-bit integer; ITA, Individual Typology Angle; LSAS, Lesion-Saliency Alignment Score; MCC, Matthews correlation coefficient; MEL, melanoma; NEV, nevus (PAD-UFES-20 label mapped to NV); NLL, negative log-likelihood; NV, melanocytic nevus; OLS, ordinary least squares; PR-AUC, area under the precision–recall curve; PTQ, post-training quantization; QAT, quantization-aware training; SCC, squamous cell carcinoma (PAD-UFES-20 class, excluded from mapping); SEK, seborrheic keratosis (PAD-UFES-20 label mapped to BKL); STS, stochastic saliency stability; TS, temperature scaling; VASC, vascular lesion; XAI, explainable artificial intelligence.")

# ---------------------------------------------------------- body
ms_static.intro(M)
M.fig(FIG / 'fig1_trust_dimensions.png', 'Consolidated architecture and workflow of the pipeline, from the two source datasets through five-seed training, Grad-CAM++ generation and the four explanation-reliability audits to the multi-seed trust report.', 15.5)
ms_static.related(M)
ms_static.methods(M, None, after_data=RC.dataset_table)
M.h(1, '4. Results')
agg2, dfc, cvv = RC.cls_sections(M)
M.fig(FIG / 'fig2_training_curves.png', 'Training curves of all five seeds (colours) with the 100-epoch sensitivity run (dashed); stars mark the epoch selected by validation macro-F1. All schedules are fully annealed.')
M.fig(FIG / 'fig3_performance_calibration.png', 'Per-class recall and F1 (mean ± SD over seeds, dots = individual seeds), row-normalized confusion matrix and reliability diagram for the reference seed (42).')
RX.sec_localization(M)
M.fig(FIG / 'fig5_correct_vs_misclassified.png', 'Distributions of LSAS, excess LSAS, calibrated confidence and lesion-area fraction for correctly classified and misclassified test images (reference seed).', 15)
M.fig(FIG / 'fig10_gallery.png', 'Examples (reference seed): top row, dermoscopic image; bottom row, Grad-CAM++ overlay with lesion outline (white). Columns show images with the highest and lowest excess LSAS among small and large lesions, for correct and misclassified predictions.', 17)
RX.sec_size(M)
M.fig(FIG / 'fig4_lesion_size_audit.png', 'Lesion-size audit (reference seed). Left: observed LSAS against lesion area with binned medians of the observed values, the shuffled-saliency null, the toroidal-shift null, the resolution-limited ceiling and the uniform expectation. Middle: excess LSAS (observed − shuffled null). Right: size strata compared with null and ceiling.', 17)
RX.sec_faith(M)
M.fig(FIG / 'fig6_faithfulness.png', 'Deletion and insertion curves (mean image baseline, reference seed; bands = 95% CI over images) for Grad-CAM++ order, random order and oracle-mask orders, and relation of excess LSAS to the deletion and insertion gains.', 17)
RX.sec_stability(M)
M.fig(FIG / 'fig7_stability.png', 'Stochastic saliency stability (300 images, 15 MC-dropout passes, reference seed) and its relation to excess LSAS.', 12)
RX.sec_skin(M)
M.fig(FIG / 'fig8_skin_tone.png', 'ITA distribution over all 10,015 HAM10000 images for the two proxy definitions, and LSAS and excess LSAS by ITA proxy stratum for all 990 test images (reference seed).', 17)
RX.sec_external(M)
ms_adapt.sec_adaptation(M, PA)
M.fig(FIG / 'fig9_pad_decomposition.png', 'Left: stepwise accuracy decomposition from HAM10000 to PAD-UFES-20 (mean ± SD over seeds; dots = seeds). Right: adaptation ladder on PAD-UFES-20 (balanced accuracy on evaluation patients).', 16)
RP.sec_sota(M)
RP.sec_null_faith(M)
M.fig(FIG / 'fig11_null_faithfulness.png', 'Null-controlled faithfulness (reference seed unless noted). Left two panels: distribution of excess insertion gain (EIG) and excess deletion gain (EDG). Third panel: excess insertion gain against excess LSAS, both null-controlled, with binned medians. Fourth panel: EIG and EDG across all five seeds.', 17)
MD.sec_architecture(M)
MD.sec_int8(M)
MD.sec_before_after(M)
MD.sec_three(M)
if ms_gui.available(): ms_gui.sec_gui(M, CF['dn_fp32']['pad_acc']['mean'])

# ---------------------------------------------------------- discussion
E5 = np.mean([PA[s]['E_PAD_mapped_5way_output']['acc'] for s in PA]); C5 = np.mean([PA[s]['C_HAM_test_5class_images_5way_output']['acc'] for s in PA])
M.h(1, '5. Discussion')
M.h(2, '5.1 Principal findings')
M.p(f"The audit supports four main findings (claims C3-C7, C10 and C12-C15 of [[T:claims]]). First, the raw association between lesion size and LSAS does not survive a geometric control: the model's own maps re-paired with other images' masks reproduce the observed size gradient almost exactly, and what remains is a small excess that decreases with lesion size; the confound replicates for ResNet50. Second, localization, faithfulness and confidence are related but distinct: raw LSAS did not predict perturbation-based faithfulness and area-adjusted localization did so only moderately; for DenseNet121 with Grad-CAM++ the null-controlled metric of Section 4.11 shows image-specific functional information that did not replicate for ResNet50 (Section 4.12), so it is a finding about a model and an explanation layer, not a general law. Third, the PAD-UFES-20 drop is large, mostly not attributable to the class mapping and consistent with external-validation losses documented elsewhere (Section 5.8); adapting dense block 4 and the head on 30% of the target patients recovered accuracy to {AD['ft_acc']:.3f} ([[T:pad_ft]]), so the zero-shot figure is not a limit of the architecture but is not a performance that can be claimed on clinical photographs. Fourth, the best configuration depends on the criterion: {S3['ties']} of {S3['n_crit']} criteria were ties under the decision rule of Section 3.15, and INT8 versions are smaller and faster and change accuracy by {abs(dacc['dn']):.1f} (DenseNet121) and {abs(dacc['rn']):.1f} (ResNet50) percentage points (Sections 4.13-4.15).")
M.h(2, '5.2 Lesion size and the interpretation of mask-overlap metrics')
M.p(f"Mask-overlap ('relevance mass') metrics are common in explanation benchmarks [@arras2022], but their dependence on target size is easily overlooked. The audit separates an expectation term (LSAS ≈ area under location-independent saliency), a ceiling term (a 7 × 7 CAM layer limits attainable LSAS for small lesions to about {sm['oracle']['med_small']:.2f}) and a residual term (excess over a null preserving the model's spatial prior); only the residual is informative about the model, and it is small. Raw LSAS therefore cannot show that small lesions are explained worse, and using it as a deferral or display-suppression criterion would flag small lesions for reasons unrelated to model behaviour. A defensible practice is to report LSAS with a matched null (cross-image shuffle preserving the spatial prior) and an area-independent statistic; the shift null, which removes centre bias, yields a larger excess than the shuffled null, so both are reported, but the conclusion about the raw association does not depend on the choice.")
M.h(2, '5.3 Localization, faithfulness and confidence')
M.p(f"Deletion and insertion ask whether the marked patches matter to the prediction. Insertion of Grad-CAM++-ranked patches beat random insertion in {frac_ig.min() * 100:.0f}–{frac_ig.max() * 100:.0f}% of images in every seed, but deletion of the top-ranked patches was less damaging than random deletion in {int((tau('faithfulness', 'mean', 'DG', 'median') < 0).sum())} of {K} seeds (median deletion gain {tau('faithfulness', 'mean', 'DG', 'median').min():+.2f} to {tau('faithfulness', 'mean', 'DG', 'median').max():+.2f}) and was negative also with a blurred baseline in the reference seed. Such disagreement is a recognised property of removal-based metrics, which reflect off-distribution artefacts as much as importance [@hooker2019,rong2022]; the evidence is therefore moderate on insertion, absent on deletion, and dependent on the training run and baseline. The oracle-mask orderings are cleaner: the predicted-class probability depends far more on lesion than on background patches. Raw LSAS, which tracks lesion size, correlated negatively with the deletion gain and inconsistently with the insertion gain; only after removing geometry did localization track insertion faithfulness (ρ = {rho_e_ig.min():.2f}–{rho_e_ig.max():.2f}). A heatmap with a high LSAS would therefore convey a false sense of functional support, and a high confidence almost none about localization.")
M.h(2, '5.4 Misclassified cases and selection bias')
M.p(f"Including misclassified predictions removed the main selection concern: LSAS and its dependence on area did not differ between correct and incorrect predictions, and confidence remained the main signal of correctness. Area-insensitive localization was modestly worse for wrong predictions in some seeds (significant in {int((tau('correct_vs_incorrect', 'loc_auc', 'p') < 0.05).sum())} of {K}) and insertion faithfulness was lower in {int((tau('faithfulness', 'mean', 'IG_corr_vs_incorr', 'p') < 0.05).sum())} of {K} seeds, but these effects are small and cannot serve as error detectors. The explained class was the predicted class, so for misclassified images the maps show what the model acted on, not what it should have looked at.")
M.h(2, '5.5 Skin-tone proxy')
M.p(f"No consistent association was found between ITA-proxy tone and area-adjusted localization: the raw difference follows lesion geometry, area-adjusted correlations ranged from ρ = {rho_e_ita.min():+.2f} to {rho_e_ita.max():+.2f} across seeds with both signs, and stratum differences reached nominal significance in {kwe_ita_sig} of {K} seeds. This is neither positive nor reassuring evidence, and the instability warns against single-run subgroup conclusions. The darkest stratum is at most a few percent of HAM10000, definition-dependent and powered only for large effects, and public datasets with darker skin are clinical photographs without lesion masks that would confound tone with modality (PAD-UFES-20 offers fewer than ten Fitzpatrick V–VI images after mapping). A meaningful fairness assessment needs new, clinician-labelled dermoscopic data with masks across skin tones; the proxy analysis shows only that no large, consistent tone-dependent artefact was found [@daneshjou2022,groh2024,wen2022].")
M.h(2, '5.6 External generalization')
M.p(f"The drop on PAD-UFES-20 is not an artefact of which classes were compared: the like-for-like five-class accuracy on HAM10000 ({C5:.3f}) is essentially the seven-class accuracy, prevalence accounts for a limited part, and the remainder ({E5:.3f} observed) reflects a shift the source-trained decision layer does not handle. A near-perfect domain classifier, an approximately four-fold larger Fréchet distance and the recovery achieved by a linear probe are consistent with a large appearance shift that leaves class information in the features, and counting absent classes as zeros in macro-F1 inflated the apparent drop. The data cannot separate acquisition device, illumination, population and label semantics (ACK vs AKIEC, SEK vs BKL); any single-cause statement would be a hypothesis. The adaptation results ([[T:pad_ft]]) show that the shift is largely recoverable with a small labelled target sample (accuracy {AD['ft_acc']:.3f} (SD {AD['ft_acc_sd']:.3f}) against {AD['zs_acc']:.3f} (SD {AD['zs_acc_sd']:.3f}) zero-shot on the same evaluation patients), so the zero-shot figure is neither the limit of the architecture nor a performance to be quoted for clinical photographs; the adapted figure is a within-dataset research estimate, not a measure at a new site. Internal accuracy and calibrated confidence do not transfer, and an explanation validated in one domain has no demonstrated validity in another.")
M.h(2, '5.7 Relation to previous work and practical implications')
M.p("The study is aligned with work arguing that trustworthy medical AI requires more than accuracy and that saliency should be evaluated rather than displayed [@ghassemi2021,rudin2019,samek2017,hauser2022,chanda2024]. Its contributions are methodological (Section 1.5), and no component (DenseNet121, Grad-CAM++, MC dropout, temperature scaling) is claimed as novel. The supported implications are limited: report localization with a null and an area-independent statistic; do not use raw LSAS thresholds for display suppression or deferral; validate calibration and explanations in the target domain before any use; and treat 'no significant difference' in small subgroups as inconclusive. Clinical utility, reader benefit and safety were not assessed.")
RP.sec_discussion_extra(M)
MD.sec_discussion_models(M)

M.h(1, '6. Study scope and methodological considerations')
M.bullets([
    "__Single split, two architectures.__ Five seeds per architecture capture training stochasticity, not data-split variability. DenseNet121 received the full eight-dimension audit; ResNet50 was audited for classification, calibration, external performance, localization geometry, faithfulness and stability, but not the skin-tone proxy or lesion-size strata. Both used Grad-CAM++ at one final convolutional layer; other architectures, methods and layers were not studied, effect magnitudes are model- and layer-specific, and the 7 × 7 target layer sets a small-lesion ceiling derived for DenseNet121 and not re-derived for ResNet50.",
    "__Calibration and convergence.__ Temperature scaling fitted on 503 images did not improve test ECE, and checkpoint selection on 502 validation images is noisy for the rarest classes; both conclusions apply to this model and these sets.",
    "__Masks.__ Masks are single-annotator and semi-automatically curated, with boundary uncertainty greatest for small lesions; Grad-CAM++ maps are coarse and only the predicted-class explanation was systematically analysed.",
    "__Faithfulness.__ Deletion/insertion metrics are baseline- and patch-size-dependent and have off-distribution artefacts; only 16 × 16 patches and two baselines were used (the blur baseline on the reference seed only). Faithfulness here is functional agreement between map and model, not clinical validity.",
    "__Skin tone.__ The ITA proxy is image-derived, unstable across two definitions for the darkest stratum, and darker-proxy images are few; no independent dermoscopic dataset with masks and darker skin was available, so fairness cannot be assessed.",
    "__External validation.__ PAD-UFES-20 differs in modality, population and label semantics; the mapping is imperfect; images above 1,024 px were downscaled before hair removal; the decomposition is descriptive and does not identify individual causes; the adaptation experiment ([[T:pad_ft]]) is within PAD-UFES-20 (adaptation and evaluation patients come from the same dataset), covers partial fine-tuning only and is an optimistic research estimate for a new site.",
    "__Null-controlled faithfulness.__ The headline excess insertion/deletion gain (Section 4.11) uses three donor images per target image and the mean-image baseline; a sensitivity check (Section 4.11, [[T:sens]]) showed the positive DenseNet121 result holds from 1 to 20 donors, with the blurred baseline and under same-class, different-class and area-matched donor selection, but it inherits the off-distribution and patch-size dependence of the metrics it extends. More consequentially, a five-seed check on ResNet50 (Section 4.12) did not reproduce a consistent positive result, so it should not be assumed to hold beyond DenseNet121 with Grad-CAM++ at the layer used here.",
    "__INT8 deployment.__ One toolchain (PyTorch FX quantization, x86 CPU kernels), one CPU and one calibration set of 503 validation images were used; the configuration was selected on the seed-42 models and applied unchanged to the other seeds; latency and tolerance may differ on other hardware or backends. Explanations were not recomputed on quantized networks (gradients are undefined through integer kernels), so the explanation findings apply to the FP32 models only.",
    "__Model comparison.__ Five seeds per architecture detect only large differences between architectures (Section 3.16), and the decision rule of Section 3.15 computes no composite score, so no overall ranking is claimed.",
    "__Scope.__ Retrospective computational study on public data (two de-identified clinical images are shown in [[F:gui]] and [[F:gui2]] only as illustrations of the interface output and contribute to no result); no reader study, no prospective validation and no patient-level split verification for HAM10000. Nothing here supports clinical deployment.",
])
M.h(1, '7. Reproducibility and Reporting')
M.p("__Code and configuration.__ All scripts are in the project folder model/ (common.py: data, model and loss definitions, including ResNet50; prep_cache.py, prep_pad.py: preprocessing; train.py: training, with an --arch flag; eval_seed.py, sts_fix.py, null_faithfulness.py, null_faith_sensitivity.py: per-seed evaluation; linear_probe_baseline.py: frozen-feature baseline; int8_*.py, bench_efficiency.py: INT8 quantization, validation-only sweeps, sensitivity analysis, quantization-aware pilot, evaluation and timing; analysis_*.py, compare_models.py, verify_statistics.py, domain_gap.py: analyses and independent verification of the summary statistics; pad_finetune.py: partial fine-tuning on PAD-UFES-20, run on the CPU; make_*.py, build_manuscript.py: figures and this manuscript). Software: Python 3.12, PyTorch 2.7.1 (CUDA 11.8 for training), torchvision, pytorch-grad-cam 1.5.x, albumentations 2.0.8, scikit-learn 1.8, SciPy, statsmodels 0.14. Hardware: one NVIDIA RTX 4050 Laptop GPU (6 GB), mixed precision for training; INT8 inference on an Intel Core i5-13450HX CPU (AVX2, six threads). Seeds 42–46; split seed 42; sampler, augmentation and dropout streams are seeded per run and explanation random baselines use seed-specific generators (1000 + seed, 2000 + seed, 500 + seed). Hyper-parameters are in Section 3.4; the checkpoint rule, 15-bin ECE, deletion/insertion protocol, MC-dropout settings (15 passes, 300 images), ITA definitions, bootstrap settings and PAD mapping are specified in Section 3. Code, per-image outputs, logits, Grad-CAM++ maps and trained weights for every seed are supplementary material and will be deposited in a public, version-controlled repository with a persistent identifier upon acceptance; a reviewer may request earlier access from the corresponding author.")
M.p("__Research interface.__ The interface of Section 4.16 is in clinical_eval_restored/: quick_predict/app.py is the single-image interface, model/model_manifest.json holds every frozen setting (checkpoint hash, class mapping, input size, normalisation, preprocessing version, calibration, Grad-CAM++ layer, software versions), and tests/ holds its automated tests; make_gui_example.py regenerates [[F:gui]], [[F:gui2]] and out/results/gui_example.json; the photographs of [[F:exam1]] and [[F:exam2]] were de-identified (face blur, removal of a stamped location band) before inclusion.")
M.p("__Data availability.__ HAM10000 and its expert lesion segmentations are available from the Harvard Dataverse (doi:10.7910/DVN/DBW86T) [@ham10000,tschandl2020]; PAD-UFES-20 is available from Mendeley Data (doi:10.17632/zr7vgbcyr2.1) [@pad]. The analyses used no patient-level data other than these public datasets; two de-identified clinical images appear in [[F:gui]] and [[F:gui2]] only as illustrations of the interface output.")
M.h(1, '8. Conclusion')
M.p(f"Three configurations were compared on one fixed lesion-level HAM10000 split with five seeds each: DenseNet121 with Grad-CAM++ (accuracy {CF['dn_fp32']['acc']['mean']:.3f}, macro-AUROC {CF['dn_fp32']['auroc']['mean']:.3f}), ResNet50 with Grad-CAM++ (accuracy {CF['rn_fp32']['acc']['mean']:.3f}, macro-AUROC {CF['rn_fp32']['auroc']['mean']:.3f}) and INT8 versions of both. No configuration was best on every criterion ([[T:three]], [[F:bars]]); the choice depends on whether accuracy, class-balanced performance, external robustness, explanation image-specificity or compute is the constraint. INT8 quantization shrank the models {shrink['dn']:.1f}- and {shrink['rn']:.1f}-fold and sped up CPU inference {speedup['dn']:.1f}- and {speedup['rn']:.1f}-fold, changing accuracy by {abs(dacc['dn']):.1f} and {abs(dacc['rn']):.1f} percentage points, but only with validation-selected settings (first convolution and head in FP32). The null-controlled audit shows that trust-related properties are not interchangeable: the steep dependence of LSAS on lesion size is a geometric property of the metric and of coarse CAM resolution, not evidence that small lesions are explained worse, and it replicates across architectures; localization, faithfulness, confidence and stability capture different things; the excess insertion/deletion gain separates an explanation from that of a different image for DenseNet121 but not ResNet50; the skin-tone result is inconclusive; and the large external drop on PAD-UFES-20, which falls below the accuracy of always predicting the most frequent class, is mostly not due to class mapping or prevalence and is largely recovered by adaptation on a small labelled target sample (a within-dataset research estimate, not a clinical validation), so target-domain evaluation and adaptation are needed. Trustworthy dermatological AI should report calibrated confidence, null-controlled localization, faithfulness with explicit baselines, target-domain validation and, where compression is used, a validated quantization configuration, and should not present accuracy, localization or a non-significant subgroup test as a substitute. Clinician-labelled dermoscopic datasets with masks across skin tones, reader studies and prospective evaluation remain necessary before any statement about clinical utility.")
M.h(1, 'Declarations')
M.p("**Funding:** Open access funding is provided by Parul University. No other specific funding was received for this work.")
M.p("**Competing interests:** The authors declare that they have no competing interests.")
M.p("**Ethics approval and consent to participate:** The statistical and explanation analyses used only previously released, publicly available, de-identified datasets (HAM10000 and PAD-UFES-20). The clinical evaluation was conducted with institutional authorization from Parul University and in collaboration with a qualified dermatologist (MBBS, MD). The study was approved by the competent university authority before commencement. All clinical images and associated information were de-identified and used exclusively for research purposes. Participant privacy and confidentiality were strictly maintained throughout data collection, analysis, and reporting.")
M.p("**Consent for publication:** Not applicable to the public datasets. The clinical images shown in [[F:gui]] and [[F:gui2]] were de-identified, as stated above.")
M.p("**Availability of data and materials:** The datasets analysed in this study are publicly available; sources and identifiers are given in Section 7.")
M.p("**Code availability:** Code, trained model weights (FP32 and INT8), per-image outputs, logits and Grad-CAM++ maps are provided as supplementary material and will be deposited in a public repository with a persistent identifier upon acceptance (Section 7).")
M.p("**Author contributions:** All authors contributed to study conception and design, data curation, model development and analysis. Soumya Das prepared the manuscript, and all authors reviewed and approved the final manuscript.")
M.h(1, 'References')
for line in M.refs_list(): M.p(line)

# ---------------------------------------------------------- verification: cross-references, captions, author metadata, declarations
import re as _re
VER = {}
xprobs = M.check_xrefs(); VER['xref_problems'] = xprobs
for pr in xprobs: check(False, 'cross-reference: ' + pr)
caps = [it[3] for it in M.items if it[0] == 't'] + [it[2] for it in M.items if it[0] == 'f']
badcap = [c[:60] for c in caps if not (_re.match(r'^(Table|Figure) \d+\. [A-Z]', c) and c.rstrip().endswith('.'))]
VER['caption_problems'] = badcap; check(not badcap, 'caption format: ' + '; '.join(badcap))
plain = [it[1] for it in M.items if it[0] == 'p']
authors = next(p for p in plain if p.startswith('Soumya Das'))
SUP = {'¹': 1, '²': 2, '³': 3, '⁴': 4, '⁵': 5, '⁶': 6}
names = [x.strip() for x in authors.split(',') if x.strip()]
mark = lambda t: {SUP[c] for c in t if c in SUP}
affil_marks = set().union(*[mark(p[:12]) for p in plain if p[:1] in SUP and 'Department' in p]) if any(p[:1] in SUP and 'Department' in p for p in plain) else set()
author_marks = set().union(*[mark(n) for n in names])
corr = [p for p in plain if p.startswith('*Corresponding author')]
email_ok = bool(corr) and bool(_re.search(r'[\w.+-]+@[\w-]+(\.[\w-]+)+', corr[0]))
VER['authors'] = dict(n_authors=len(names), author_markers=sorted(author_marks), affiliation_markers=sorted(affil_marks), corresponding_marker_count=authors.count('*'), corresponding_line=bool(corr), email_ok=email_ok)
check(len(names) == 6, f'author count is {len(names)}, expected 6')
check(author_marks == {1, 2, 3, 4, 5, 6} and affil_marks == {1, 2, 3, 4, 5, 6}, 'author/affiliation numbering does not cover 1-6 consistently')
check(authors.count('*') == 1 and bool(corr) and email_ok, 'corresponding-author marker/line/email problem')
decl_i = next(i for i, it in enumerate(M.items) if it[0] == 'h' and it[2] == 'Declarations')
dtxt = ' '.join(it[1] for it in M.items[decl_i + 1:decl_i + 12] if it[0] == 'p')
need = ['Funding', 'Competing interests', 'Ethics approval and consent to participate', 'Consent for publication', 'Availability of data and materials', 'Code availability', 'Author contributions']
VER['declarations_missing'] = [n for n in need if n not in dtxt]; check(not VER['declarations_missing'], 'declarations missing: ' + ', '.join(VER['declarations_missing']))
json.dump(VER, open(RES / 'submission_checks.json', 'w'), indent=1)
M.to_docx(OUT / 'Revised_Manuscript_SkinCancer_ICCSDI2026.docx'); open(OUT / 'Revised_Manuscript_SkinCancer_ICCSDI2026.md', 'w', encoding='utf-8').write(M.to_markdown())
json.dump(WARN, open(RES / 'claim_check_warnings.json', 'w'))
print('built; references:', len(M.order), 'tables:', M.tab_n, 'figures:', M.fig_n, 'warnings:', WARN)
