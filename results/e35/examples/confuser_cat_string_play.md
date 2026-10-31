# Behaviour question (cat)

**Scene (owner):** Living room floor with a play mat.  
**Question (owner):** My cat keeps turning in circles while chasing or biting its tail. Why does it do this?

**Answer from:** local (strategy local_preprocessed; cloud status unavailable)

Packet: 43 images, 0.56 MB upload, about 10349 input tokens (estimate 0.04605 USD)

## On-device / local answer

**Species (given):** cat  
**Scene:** Living room floor with a play mat.  
**Question:** My cat keeps turning in circles while chasing or biting its tail. Why does it do this?

**Summary:** In the recording: rolling on the back or rubbing the face or body on objects (1.5-27.5 s); batting, pouncing on or chasing a toy or object (11.0-11.5 s, 18.0-20.0 s). Movement: the animal stays roughly in place while moving. Turning in tight circles (after its own tail or not) (user report only) has several possible explanations, in order of fit to your description: discomfort, itch, pain or a neurological cause; ordinary play or maintenance behaviour; arousal, frustration or stress. No request is implied by it. Licking, biting or chewing its own tail (user report only) has several possible explanations, in order of fit to your description: discomfort, itch, pain or a neurological cause; ordinary play or maintenance behaviour; an abnormal repetitive (compulsive) pattern. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- rolling_rubbing (score 0.92, threshold 0.30; bouts [[1.5, 27.5]])
- playing_object (score 0.37, threshold 0.34; bouts [[11.0, 11.5], [18.0, 20.0]])

**What you reported:**
- tail_chasing: not recognized in the recording (it may be outside the clip, brief, or missed by the recognizer)
- tail_directed: not recognized in the recording (it may be outside the clip, brief, or missed by the recognizer)

**Possible explanations for tail_chasing** (user report only; kinds in order of fit to your description: medical, play_or_normal, arousal_or_stress):
- [medical] itch or skin disease at the tail base (fleas, allergy, other dermatitis) - one reason a cat turns to bite or lick the tail (sources: amengualbatle2019, tvp_fhs, vin_pa; matched in your description: bit, biting)
- [medical] pain in the tail, the lower back or the anal sacs (sources: tvp_fhs, cornell_fhs; matched in your description: -)
- [medical] feline hyperaesthesia syndrome: brief episodes with rippling skin over the lower back, dilated pupils, tail chasing or biting, sudden running and sometimes vocalizing - a diagnosis made after other causes are excluded (sources: cornell_fhs, amengualbatle2019, tvp_fhs, vrc2026_fhs_cat; matched in your description: -)
- [medical] focal seizures, when episodes are abrupt, stereotyped and cannot be interrupted (sources: tvp_fhs; matched in your description: -)
- [medical] circling that is not tail chasing: tight circles always in the same direction, often with a head tilt or loss of balance, point to a balance (vestibular) or brain problem (sources: merck_neuro_exam, dvm360_vestibular, vin_vestibular; matched in your description: -)
- [play_or_normal] predatory play with the tail as a moving target, typical of kittens and playful young cats (sources: mrfrs_vet_tail, ellis2013; matched in your description: -)
- [arousal_or_stress] stress or conflict (a household change, other pets); it can add to the episodes but is not shown to cause them (sources: cornell_fhs, vrc2026_fhs_cat; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for tail_directed** (user report only; kinds in order of fit to your description: medical, play_or_normal, compulsive):
- [medical] itch or skin disease (parasites, allergy) (sources: waisglass2006, vin_pa, amengualbatle2019; matched in your description: bit)
- [medical] pain or a lesion near the tail or anal sacs; in dogs after surgery, neuropathic pain was a differential (sources: tvp_fhs, frontiers2023_dachshund; matched in your description: -)
- [play_or_normal] ordinary grooming of the tail as part of a grooming sequence (sources: vin_pa; matched in your description: -)
- [compulsive] a compulsive or displacement pattern when medical causes are excluded (sources: cliniciansbrief_pa, tiira2012; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for rolling_rubbing** (recording; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] a response to catnip or other cat-attracting plants (silver vine, honeysuckle, valerian): rolling, rubbing, licking and chewing; most cats respond to at least one of these plants (sources: bol2017; matched in your description: -)
- [play_or_normal] ordinary scent marking or greeting by rubbing the face on objects or people (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for playing_object** (recording; kinds in order of fit to your description: play_or_normal, request):
- [play_or_normal] predatory or object play; cats need opportunities for play and predatory behaviour (sources: ellis2013; matched in your description: -)
- [request] an invitation for interactive play when the animal brings the toy to a person or looks back at them (sources: ellis2013; matched in your description: -)
- request or intention possible: yes

**Context observed in the recording (automatic, can be wrong):**
- a person in view (detector, score 0.71; [[3.0, 3.2], [17.8, 18.8], [20.0, 21.8]] s)
- people talking (sound model (YAMNet), score 0.5; [[26.4, 27.4]] s)
- place: veterinary clinic (scene prompts (SigLIP), score 0.56)

**Before and after:**
- rolling_rubbing 1.5-27.5 s: before - nothing recognized; after - nothing recognized
- playing_object 11.0-11.5 s: before - nothing recognized; after - nothing recognized
- playing_object 18.0-20.0 s: before - nothing recognized; after - a person in view (20.0-21.8 s)

**Would help tell them apart:**
- How old is the cat, and is the tail chasing new or has it done this since kittenhood?
- Does the skin on the lower back ripple or twitch, or do the pupils dilate, during an episode? Does the cat run off afterwards?
- Does the cat react when you touch the base of the tail or the lower back?
- Is there hair loss, scabs, flea dirt or a wound on the tail or near its base?
- Can you interrupt an episode (call, toy), and how long does it last?
- Does it always circle in the same direction, with a head tilt or loss of balance?
- Is there hair loss, redness, a wound or flea dirt at the tail?
- Did it start after an injury, surgery or a change at home?

**See a veterinarian promptly if:**
- the tail is wounded, bleeding or the cat keeps attacking it
- episodes with unresponsiveness, drooling or loss of bladder control
- a sudden new behaviour in an adult or senior cat
- circling with a head tilt, falling or loss of balance
- open wounds or bleeding
- the animal cannot be distracted from damaging the tail

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [amengualbatle2019] Amengual Batle P et al. (2019) Feline hyperaesthesia syndrome with self-trauma to the tail: retrospective study of seven cases. J Feline Med Surg 21(2). (peer_reviewed_study) https://pmc.ncbi.nlm.nih.gov/articles/PMC10814607/
- [tvp_fhs] Today's Veterinary Practice: Answering FAQs about feline hyperesthesia syndrome. (clinical_review) https://todaysveterinarypractice.com/behavior/answering-faqs-about-feline-hyperesthesia-syndrome/
- [vin_pa] Veterinary Partner (VIN): Psychogenic alopecia. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&catId=254060&id=11924460
- [mrfrs_vet_tail] Merrimack River Feline Rescue Society, Ask a Vet: Why do cats chase their tails? (veterinarian_advice) https://mrfrs.org/ask-a-vet-with-dr-sam-why-do-cats-chase-their-tails/
- [ellis2013] Ellis SLH et al. (2013) AAFP and ISFM feline environmental needs guidelines. J Feline Med Surg 15:219-230 (as summarized by Veterinary Partner). (clinical_review) https://doi.org/10.1177/1098612X13477537
- [cornell_fhs] Cornell Feline Health Center: Hyperesthesia syndrome. (veterinary_school) https://www.vet.cornell.edu/departments-centers-and-institutes/cornell-feline-health-center/health-information/feline-health-topics/hyperesthesia-syndrome
- [vrc2026_fhs_cat] A presumptive diagnosis of feline hyperesthesia syndrome due to stressful human-related condition: case report. Vet Res Commun (2026). (peer_reviewed_case_report) https://doi.org/10.1007/s11259-026-11255-8
- [merck_neuro_exam] Merck Veterinary Manual: The neurologic examination of animals. (clinical_review) https://www.merckvetmanual.com/nervous-system/the-neurologic-examination/the-neurologic-examination-of-animals
- [dvm360_vestibular] dvm360: Vestibular disease in dogs and cats (proceedings). (clinical_review) https://www.dvm360.com/view/vestibular-disease-dogs-and-cats-proceedings
- [vin_vestibular] Veterinary Partner (VIN): Vestibular disease in dogs and cats. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&id=4951444
- [waisglass2006] Waisglass SE et al. (2006) Underlying medical conditions in cats with presumptive psychogenic alopecia. J Am Vet Med Assoc 228(11):1705-1709. (peer_reviewed_study) https://avmajournals.avma.org/view/journals/javma/228/11/javma.228.11.1705.xml
- [frontiers2023_dachshund] Case report: Neuropathic pain versus undesirable behavior in a Dachshund after hemilaminectomy. Front Vet Sci (2023). (peer_reviewed_case_report) https://doi.org/10.3389/fvets.2023.1223800
- [cliniciansbrief_pa] Moffat K. Psychogenic alopecia in cats. Clinician's Brief (2012). (clinical_review) https://www.cliniciansbrief.com/article/psychogenic-alopecia-cats
- [tiira2012] Tiira K et al. (2012) Environmental effects on compulsive tail chasing in dogs. PLoS ONE 7(7): e41684. (peer_reviewed_study) https://doi.org/10.1371/journal.pone.0041684
- [bol2017] Bol S et al. (2017) Responsiveness of cats to silver vine, Tatarian honeysuckle, valerian and catnip. BMC Vet Res 13:70. (peer_reviewed_study) https://doi.org/10.1186/s12917-017-0987-6
