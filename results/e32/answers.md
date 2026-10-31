# E32 answers of the v2 system (all benchmark questions)

### t01 - archive_ia_cat_tail_3 (tail_chasing)

**Species (given):** cat  
**Scene:** Carpeted living room; we have a second cat.  
**Question:** My cat whirls around after his tail on the carpet and then flops down. Is he just playing?

**Summary:** In the recording: turning in tight circles (after its own tail or not) (2.0-6.0 s, 10.0-14.5 s, 19.5-25.0 s, 35.0-39.0 s); scratching its own body with a hind foot (0.0-2.5 s, 5.5-6.5 s, 8.5-9.0 s). Movement: the animal stays roughly in place while moving. Turning in tight circles (after its own tail or not) (recording) has several possible explanations, in order of fit to your description: ordinary play or maintenance behaviour; discomfort, itch, pain or a neurological cause; arousal, frustration or stress. No request is implied by it. Scratching its own body with a hind foot (recording) has several possible explanations, in order of fit to your description: discomfort, itch, pain or a neurological cause; ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- tail_chasing (score 0.80, threshold 0.64; bouts [[2.0, 6.0], [10.0, 14.5], [19.5, 25.0], [35.0, 39.0]])
- scratching_self (score 0.30, threshold 0.30; bouts [[0.0, 2.5], [5.5, 6.5], [8.5, 9.0]])

**What you reported:**
- no behaviour recognized in the question

**Possible explanations for tail_chasing** (recording; kinds in order of fit to your description: play_or_normal, medical, arousal_or_stress):
- [play_or_normal] predatory play with the tail as a moving target, typical of kittens and playful young cats (sources: mrfrs_vet_tail, ellis2013; matched in your description: playing)
- [medical] itch or skin disease at the tail base (fleas, allergy, other dermatitis) - one reason a cat turns to bite or lick the tail (sources: amengualbatle2019, tvp_fhs, vin_pa; matched in your description: -)
- [medical] pain in the tail, the lower back or the anal sacs (sources: tvp_fhs, cornell_fhs; matched in your description: -)
- [medical] feline hyperaesthesia syndrome: brief episodes with rippling skin over the lower back, dilated pupils, tail chasing or biting, sudden running and sometimes vocalizing - a diagnosis made after other causes are excluded (sources: cornell_fhs, amengualbatle2019, tvp_fhs, vrc2026_fhs_cat; matched in your description: -)
- [medical] focal seizures, when episodes are abrupt, stereotyped and cannot be interrupted (sources: tvp_fhs; matched in your description: -)
- [medical] circling that is not tail chasing: tight circles always in the same direction, often with a head tilt or loss of balance, point to a balance (vestibular) or brain problem (sources: merck_neuro_exam, dvm360_vestibular, vin_vestibular; matched in your description: -)
- [arousal_or_stress] stress or conflict (a household change, other pets); it can add to the episodes but is not shown to cause them (sources: cornell_fhs, vrc2026_fhs_cat; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for scratching_self** (recording; kinds in order of fit to your description: medical, play_or_normal):
- [medical] itch from parasites, allergy or ear disease when it is frequent or focused (sources: vin_pa, waisglass2006; matched in your description: -; matched in the recording: ear)
- [play_or_normal] occasional scratching is ordinary coat care (sources: vin_pa; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.45; [[12.8, 13.2]] s)
- TV or music (sound model (YAMNet), score 0.99; [[0.0, 1.0], [0.5, 1.4], [1.0, 1.9]] s)
- place: living room (scene prompts (SigLIP), score 0.67)

**Before and after:**
- tail_chasing 2.0-6.0 s: before - TV or music (0.0-1.0 s), TV or music (0.5-1.4 s), TV or music (1.0-1.9 s), TV or music (1.4-2.4 s), scratching_self (0.0-2.5 s); after - scratching_self (5.5-6.5 s), scratching_self (8.5-9.0 s)
- tail_chasing 10.0-14.5 s: before - scratching_self (5.5-6.5 s), scratching_self (8.5-9.0 s); after - nothing recognized
- scratching_self 0.0-2.5 s: before - nothing recognized; after - TV or music (2.4-3.4 s), tail_chasing (2.0-6.0 s)
- scratching_self 5.5-6.5 s: before - TV or music (0.0-1.0 s), TV or music (0.5-1.4 s), TV or music (1.0-1.9 s), TV or music (1.4-2.4 s), TV or music (1.9-2.9 s), TV or music (2.4-3.4 s); after - tail_chasing (10.0-14.5 s)

**Would help tell them apart:**
- How old is the cat, and is the tail chasing new or has it done this since kittenhood?
- Does the skin on the lower back ripple or twitch, or do the pupils dilate, during an episode? Does the cat run off afterwards?
- Does the cat react when you touch the base of the tail or the lower back?
- Is there hair loss, scabs, flea dirt or a wound on the tail or near its base?
- Can you interrupt an episode (call, toy), and how long does it last?
- Does it always circle in the same direction, with a head tilt or loss of balance?
- Is it frequent or focused on one place (ears, neck)?
- Are there fleas, flea dirt, redness or discharge from the ears?

**See a veterinarian promptly if:**
- the tail is wounded, bleeding or the cat keeps attacking it
- episodes with unresponsiveness, drooling or loss of bladder control
- a sudden new behaviour in an adult or senior cat
- circling with a head tilt, falling or loss of balance
- bleeding or open sores from scratching

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [mrfrs_vet_tail] Merrimack River Feline Rescue Society, Ask a Vet: Why do cats chase their tails? (veterinarian_advice) https://mrfrs.org/ask-a-vet-with-dr-sam-why-do-cats-chase-their-tails/
- [ellis2013] Ellis SLH et al. (2013) AAFP and ISFM feline environmental needs guidelines. J Feline Med Surg 15:219-230 (as summarized by Veterinary Partner). (clinical_review) https://doi.org/10.1177/1098612X13477537
- [amengualbatle2019] Amengual Batle P et al. (2019) Feline hyperaesthesia syndrome with self-trauma to the tail: retrospective study of seven cases. J Feline Med Surg 21(2). (peer_reviewed_study) https://pmc.ncbi.nlm.nih.gov/articles/PMC10814607/
- [tvp_fhs] Today's Veterinary Practice: Answering FAQs about feline hyperesthesia syndrome. (clinical_review) https://todaysveterinarypractice.com/behavior/answering-faqs-about-feline-hyperesthesia-syndrome/
- [vin_pa] Veterinary Partner (VIN): Psychogenic alopecia. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&catId=254060&id=11924460
- [cornell_fhs] Cornell Feline Health Center: Hyperesthesia syndrome. (veterinary_school) https://www.vet.cornell.edu/departments-centers-and-institutes/cornell-feline-health-center/health-information/feline-health-topics/hyperesthesia-syndrome
- [vrc2026_fhs_cat] A presumptive diagnosis of feline hyperesthesia syndrome due to stressful human-related condition: case report. Vet Res Commun (2026). (peer_reviewed_case_report) https://doi.org/10.1007/s11259-026-11255-8
- [merck_neuro_exam] Merck Veterinary Manual: The neurologic examination of animals. (clinical_review) https://www.merckvetmanual.com/nervous-system/the-neurologic-examination/the-neurologic-examination-of-animals
- [dvm360_vestibular] dvm360: Vestibular disease in dogs and cats (proceedings). (clinical_review) https://www.dvm360.com/view/vestibular-disease-dogs-and-cats-proceedings
- [vin_vestibular] Veterinary Partner (VIN): Vestibular disease in dogs and cats. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&id=4951444
- [waisglass2006] Waisglass SE et al. (2006) Underlying medical conditions in cats with presumptive psychogenic alopecia. J Am Vet Med Assoc 228(11):1705-1709. (peer_reviewed_study) https://avmajournals.avma.org/view/journals/javma/228/11/javma.228.11.1705.xml

### t02 - paper_sensory_compulsive_2020 (tail_chasing)

**Species (given):** dog  
**Scene:** Our dog was born deaf and partly blind; it happens in the garden and in the house.  
**Question:** My deaf white dog stares at nothing and then chases her tail for minutes. Our vet first thought it was neurological. Why does she do it?

**Summary:** In the recording: barking, whining, howling or growling (57.6-60.0 s); turning in tight circles (after its own tail or not) (7.5-10.0 s, 13.5-21.0 s, 37.0-45.0 s, 56.0-58.0 s); rhythmic bobbing, nodding or rotating of the head while awake (7.5-9.5 s, 11.5-16.5 s, 20.0-20.5 s, 46.0-46.5 s and 2 more). Movement: the animal stays roughly in place while moving. Turning in tight circles (after its own tail or not) (recording and user report) has several possible explanations, in order of fit to your description: a request or intention directed at a person; ordinary play or maintenance behaviour; arousal, frustration or stress; an abnormal repetitive (compulsive) pattern; discomfort, itch, pain or a neurological cause. Staring fixedly or snapping at the air as if catching flies (user report only) has several possible explanations, in order of fit to your description: discomfort, itch, pain or a neurological cause; ordinary play or maintenance behaviour; an abnormal repetitive (compulsive) pattern. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- vocalizing (score 0.92, threshold 0.30; bouts [[57.6, 60.0]])
- tail_chasing (score 0.64, threshold 0.64; bouts [[7.5, 10.0], [13.5, 21.0], [37.0, 45.0], [56.0, 58.0]])
- head_tremor (score 0.42, threshold 0.30; bouts [[7.5, 9.5], [11.5, 16.5], [20.0, 20.5], [46.0, 46.5]])

**What you reported:**
- tail_chasing: also seen in the recording
- staring_snapping: not recognized in the recording (it may be outside the clip, brief, or missed by the recognizer)
- head_tremor: also seen in the recording

**Possible explanations for tail_chasing** (recording and user report; kinds in order of fit to your description: request, play_or_normal, arousal_or_stress, compulsive, medical):
- [request] a learned way to get attention: owners often laugh at or encourage tail chasing, which can reinforce it; some dogs start when nobody attends to them (sources: burn2011, tiira2012; matched in your description: -; matched in the recording: talking)
- [play_or_normal] self-directed play or excitement, especially in young dogs, when bouts are brief and the dog is easy to distract (sources: burn2011, tiira2012; matched in your description: -)
- [arousal_or_stress] frustration or arousal in a waiting or conflict situation (for example waiting for food, or when the owner talks to other people) (sources: tiira2012; matched in your description: -)
- [compulsive] canine compulsive disorder: habitual, perseverative chasing with lowered responsiveness, often with other repetitive behaviours (light chasing, freezing); more common in some breeds (bull terriers, German shepherds) (sources: tiira2012, burn2011; matched in your description: -)
- [medical] a physical cause near the tail or a neurological one: tail or anal-sac discomfort, skin itch, pain, or focal seizures; behaviour and medical causes can coexist (sources: vanhaevermaet2026, frontiers2026_focal; matched in your description: -)
- [medical] circling that is not tail chasing: tight circles always in the same direction, often with a head tilt or head turn, loss of balance or episodes that cannot be interrupted, point to a balance (vestibular) or brain problem (sources: merck_neuro_exam, dvm360_vestibular, vin_vestibular; matched in your description: -)
- request or intention possible: yes

**Possible explanations for staring_snapping** (user report only; kinds in order of fit to your description: medical, play_or_normal, compulsive):
- [medical] focal (impaired-awareness) seizures: fly-catching, air snapping, staring and lip licking with reduced responsiveness, which can be mistaken for a behavioural problem (sources: frontiers2026_focal; matched in your description: nothing)
- [play_or_normal] snapping at a real insect or reacting to something actually in view (sources: none; matched in your description: -)
- [compulsive] a compulsive behaviour, the main non-medical explanation weighed for such episodes (sources: frontiers2026_focal, tiira2012; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for head_tremor** (recording and user report; kinds in order of fit to your description: medical):
- [medical] shaking the head to the side because of ear discomfort is a different movement (a vigorous shake, often with ear scratching) (sources: vin_vestibular; matched in your description: -; matched in the recording: ear)
- [medical] idiopathic head tremor: episodes of head bobbing in an alert dog that usually stop when the dog is distracted (called, offered food or a toy); considered benign but diagnosed after other causes are excluded (sources: mspca_ihts; matched in your description: -)
- [medical] a focal seizure, the main alternative: it is not stopped by distraction and may come with drooling, unresponsiveness or other signs (sources: mspca_ihts; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for vocalizing** (recording; kinds in order of fit to your description: arousal_or_stress, request, play_or_normal):
- [arousal_or_stress] alerting to a disturbance (a stranger, the doorbell) (sources: yin2004; matched in your description: -)
- [request] seeking contact when separated or isolated (sources: yin2004; matched in your description: -)
- [play_or_normal] play barking during games (sources: yin2004; matched in your description: -)
- request or intention possible: yes

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 1.0; [[7.0, 7.5], [13.2, 13.3], [43.8, 50.3]] s)
- people talking (sound model (YAMNet), score 0.5; [[9.1, 10.1]] s)
- TV or music (sound model (YAMNet), score 0.98; [[7.7, 8.6], [8.2, 9.1], [15.8, 16.8]] s)

**Before and after:**
- vocalizing 57.6-60.0 s: before - a person close to the animal (52.3-53.8 s), head_tremor (52.5-53.5 s), tail_chasing (56.0-58.0 s); after - nothing recognized
- tail_chasing 7.5-10.0 s: before - a person close to the animal (7.0-7.5 s); after - a person close to the animal (13.2-13.3 s), head_tremor (11.5-16.5 s)
- tail_chasing 13.5-21.0 s: before - TV or music (7.7-8.6 s), TV or music (8.2-9.1 s), a person close to the animal (13.2-13.3 s), head_tremor (7.5-9.5 s), people talking (9.1-10.1 s); after - nothing recognized
- head_tremor 7.5-9.5 s: before - a person close to the animal (7.0-7.5 s); after - a person close to the animal (13.2-13.3 s), people talking (9.1-10.1 s), tail_chasing (13.5-21.0 s)
- head_tremor 11.5-16.5 s: before - TV or music (7.7-8.6 s), TV or music (8.2-9.1 s), a person close to the animal (7.0-7.5 s), people talking (9.1-10.1 s), tail_chasing (7.5-10.0 s); after - TV or music (16.3-17.3 s), TV or music (16.8-17.8 s), TV or music (17.3-18.2 s)

**Would help tell them apart:**
- How old was the dog when it started, and how often does it happen (occasionally or daily)?
- Can you interrupt it by calling the dog or offering a toy, and does it respond to you during a bout?
- In which situations does it start (waiting for food, when people talk to each other, when left alone, when excited)?
- Does the dog bite or injure the tail, scoot, or show hair loss or sores near the tail base?
- Are there other repetitive behaviours such as chasing lights or shadows, snapping at the air, freezing or staring?
- Does it always circle in the same direction, with a head tilt or loss of balance, even without looking at its tail?
- Does the dog respond to its name or a touch during an episode?
- Is anything actually there (an insect, a shadow, a light)?

**See a veterinarian promptly if:**
- the tail is wounded or bleeding
- bouts cannot be interrupted or interfere with eating or sleeping
- episodes with staring, unresponsiveness or snapping at the air (possible seizures)
- circling with a head tilt, falling or loss of balance
- episodes of unresponsiveness, drooling, collapse or loss of bladder control (possible seizures)
- episodes with unresponsiveness, drooling, collapse or loss of bladder control
- a first episode in an older dog

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- The detector's species vote disagrees with the species you gave (dog); your answer is used, and the detector only locates the animal.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [burn2011] Burn CC (2011) A vicious cycle: a cross-sectional study of canine tail-chasing and human responses to it, using a free video-sharing website. PLoS ONE 6(11): e26553. (peer_reviewed_study) https://doi.org/10.1371/journal.pone.0026553
- [tiira2012] Tiira K et al. (2012) Environmental effects on compulsive tail chasing in dogs. PLoS ONE 7(7): e41684. (peer_reviewed_study) https://doi.org/10.1371/journal.pone.0041684
- [vanhaevermaet2026] Van Haevermaet et al. (2026) Tail chasing in a dog: false dichotomy between behavioural and medical problems. Veterinary Record Case Reports. (peer_reviewed_case_report) https://doi.org/10.1002/vrc2.70334
- [frontiers2026_focal] Behavioral and autonomic manifestations of suspected focal impaired awareness seizures supported by interictal EEG (three dogs). Front Vet Sci (2026). (peer_reviewed_case_report) https://doi.org/10.3389/fvets.2026.1904889
- [merck_neuro_exam] Merck Veterinary Manual: The neurologic examination of animals. (clinical_review) https://www.merckvetmanual.com/nervous-system/the-neurologic-examination/the-neurologic-examination-of-animals
- [dvm360_vestibular] dvm360: Vestibular disease in dogs and cats (proceedings). (clinical_review) https://www.dvm360.com/view/vestibular-disease-dogs-and-cats-proceedings
- [vin_vestibular] Veterinary Partner (VIN): Vestibular disease in dogs and cats. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&id=4951444
- [mspca_ihts] MSPCA-Angell: Idiopathic head tremor syndrome. (veterinary_hospital) https://www.mspca.org/clinical/idiopathic-head-tremor-syndrome/
- [yin2004] Yin S, McCowan B (2004) Barking in domestic dogs: context specificity and individual identification. Anim Behav 68:343-355. (peer_reviewed_study) https://doi.org/10.1016/j.anbehav.2003.07.016

### t03 - paper_hamartoma_circling_2025 (circling_neuro)

**Species (given):** dog  
**Scene:** At the vet clinic; she has had a head tilt to the left for months and loses her balance.  
**Question:** My 7-month-old Schnauzer keeps turning her head to the left over and over and sometimes circles to the left. She can't be distracted out of it. What could be going on?

**Summary:** In the recording: rhythmic bobbing, nodding or rotating of the head while awake (4.5-14.0 s, 18.5-19.0 s, 29.0-36.5 s); batting, pouncing on or chasing a toy or object (4.0-4.5 s, 15.0-17.0 s, 19.0-22.5 s, 26.0-29.0 s). Movement: the animal stays roughly in place while moving. Turning in tight circles (after its own tail or not) (user report only) has several possible explanations, in order of fit to your description: discomfort, itch, pain or a neurological cause; ordinary play or maintenance behaviour; a request or intention directed at a person; arousal, frustration or stress; an abnormal repetitive (compulsive) pattern. Rhythmic bobbing, nodding or rotating of the head while awake (recording) is usually discomfort, itch, pain or a neurological cause. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- head_tremor (score 0.97, threshold 0.30; bouts [[4.5, 14.0], [18.5, 19.0], [29.0, 36.5]])
- playing_object (score 0.45, threshold 0.34; bouts [[4.0, 4.5], [15.0, 17.0], [19.0, 22.5], [26.0, 29.0]])

**What you reported:**
- tail_chasing: not recognized in the recording (it may be outside the clip, brief, or missed by the recognizer)

**Possible explanations for tail_chasing** (user report only; kinds in order of fit to your description: medical, play_or_normal, request, arousal_or_stress, compulsive):
- [medical] circling that is not tail chasing: tight circles always in the same direction, often with a head tilt or head turn, loss of balance or episodes that cannot be interrupted, point to a balance (vestibular) or brain problem (sources: merck_neuro_exam, dvm360_vestibular, vin_vestibular; matched in your description: to the left, head tilt, tilt, balance, can't be distracted)
- [medical] a physical cause near the tail or a neurological one: tail or anal-sac discomfort, skin itch, pain, or focal seizures; behaviour and medical causes can coexist (sources: vanhaevermaet2026, frontiers2026_focal; matched in your description: -)
- [play_or_normal] self-directed play or excitement, especially in young dogs, when bouts are brief and the dog is easy to distract (sources: burn2011, tiira2012; matched in your description: -)
- [request] a learned way to get attention: owners often laugh at or encourage tail chasing, which can reinforce it; some dogs start when nobody attends to them (sources: burn2011, tiira2012; matched in your description: -)
- [arousal_or_stress] frustration or arousal in a waiting or conflict situation (for example waiting for food, or when the owner talks to other people) (sources: tiira2012; matched in your description: -)
- [compulsive] canine compulsive disorder: habitual, perseverative chasing with lowered responsiveness, often with other repetitive behaviours (light chasing, freezing); more common in some breeds (bull terriers, German shepherds) (sources: tiira2012, burn2011; matched in your description: -)
- request or intention possible: yes

**Possible explanations for head_tremor** (recording; kinds in order of fit to your description: medical):
- [medical] idiopathic head tremor: episodes of head bobbing in an alert dog that usually stop when the dog is distracted (called, offered food or a toy); considered benign but diagnosed after other causes are excluded (sources: mspca_ihts; matched in your description: distract)
- [medical] shaking the head to the side because of ear discomfort is a different movement (a vigorous shake, often with ear scratching) (sources: vin_vestibular; matched in your description: -; matched in the recording: ear)
- [medical] a focal seizure, the main alternative: it is not stopped by distraction and may come with drooling, unresponsiveness or other signs (sources: mspca_ihts; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for playing_object** (recording; kinds in order of fit to your description: play_or_normal, request):
- [play_or_normal] predatory or object play; cats need opportunities for play and predatory behaviour (sources: ellis2013; matched in your description: -)
- [request] an invitation for interactive play when the animal brings the toy to a person or looks back at them (sources: ellis2013; matched in your description: -)
- request or intention possible: yes

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.98; [[5.3, 5.7], [19.2, 27.2], [28.8, 36.2]] s)
- place: veterinary clinic (scene prompts (SigLIP), score 0.99)

**Before and after:**
- head_tremor 4.5-14.0 s: before - playing_object (4.0-4.5 s); after - playing_object (15.0-17.0 s), playing_object (19.0-22.5 s)
- head_tremor 18.5-19.0 s: before - playing_object (15.0-17.0 s); after - a person close to the animal (19.2-27.2 s), playing_object (19.0-22.5 s)
- playing_object 4.0-4.5 s: before - nothing recognized; after - a person close to the animal (5.3-5.7 s), head_tremor (4.5-14.0 s)
- playing_object 15.0-17.0 s: before - head_tremor (4.5-14.0 s); after - a person close to the animal (19.2-27.2 s), head_tremor (18.5-19.0 s)

**Would help tell them apart:**
- How old was the dog when it started, and how often does it happen (occasionally or daily)?
- Can you interrupt it by calling the dog or offering a toy, and does it respond to you during a bout?
- In which situations does it start (waiting for food, when people talk to each other, when left alone, when excited)?
- Does the dog bite or injure the tail, scoot, or show hair loss or sores near the tail base?
- Are there other repetitive behaviours such as chasing lights or shadows, snapping at the air, freezing or staring?
- Does it always circle in the same direction, with a head tilt or loss of balance, even without looking at its tail?
- Does the dog stay alert, and does the movement stop when you call it or offer a treat?
- Is it a small rhythmic bobbing (yes-yes, no-no) or a vigorous shake as if shaking off water or an ear problem?

**See a veterinarian promptly if:**
- the tail is wounded or bleeding
- bouts cannot be interrupted or interfere with eating or sleeping
- episodes with staring, unresponsiveness or snapping at the air (possible seizures)
- circling with a head tilt, falling or loss of balance
- episodes with unresponsiveness, drooling, collapse or loss of bladder control
- a first episode in an older dog

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [merck_neuro_exam] Merck Veterinary Manual: The neurologic examination of animals. (clinical_review) https://www.merckvetmanual.com/nervous-system/the-neurologic-examination/the-neurologic-examination-of-animals
- [dvm360_vestibular] dvm360: Vestibular disease in dogs and cats (proceedings). (clinical_review) https://www.dvm360.com/view/vestibular-disease-dogs-and-cats-proceedings
- [vin_vestibular] Veterinary Partner (VIN): Vestibular disease in dogs and cats. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&id=4951444
- [burn2011] Burn CC (2011) A vicious cycle: a cross-sectional study of canine tail-chasing and human responses to it, using a free video-sharing website. PLoS ONE 6(11): e26553. (peer_reviewed_study) https://doi.org/10.1371/journal.pone.0026553
- [tiira2012] Tiira K et al. (2012) Environmental effects on compulsive tail chasing in dogs. PLoS ONE 7(7): e41684. (peer_reviewed_study) https://doi.org/10.1371/journal.pone.0041684
- [vanhaevermaet2026] Van Haevermaet et al. (2026) Tail chasing in a dog: false dichotomy between behavioural and medical problems. Veterinary Record Case Reports. (peer_reviewed_case_report) https://doi.org/10.1002/vrc2.70334
- [frontiers2026_focal] Behavioral and autonomic manifestations of suspected focal impaired awareness seizures supported by interictal EEG (three dogs). Front Vet Sci (2026). (peer_reviewed_case_report) https://doi.org/10.3389/fvets.2026.1904889
- [mspca_ihts] MSPCA-Angell: Idiopathic head tremor syndrome. (veterinary_hospital) https://www.mspca.org/clinical/idiopathic-head-tremor-syndrome/
- [ellis2013] Ellis SLH et al. (2013) AAFP and ISFM feline environmental needs guidelines. J Feline Med Surg 15:219-230 (as summarized by Veterinary Partner). (clinical_review) https://doi.org/10.1177/1098612X13477537

### t04 - stock_mixkit_45868 (confuser)

**Species (given):** dog  
**Scene:** In the park with a tennis ball.  
**Question:** In the park my young corgi races after his tennis ball and runs back to me. Does he want me to keep throwing it?

**Summary:** In the recording: rhythmic bobbing, nodding or rotating of the head while awake (0.0-3.5 s); batting, pouncing on or chasing a toy or object (3.5-4.0 s). Movement: the animal stays roughly in place while moving. Rhythmic bobbing, nodding or rotating of the head while awake (recording) is usually discomfort, itch, pain or a neurological cause. No request is implied by it. Batting, pouncing on or chasing a toy or object (recording) has several possible explanations, in order of fit to your description: ordinary play or maintenance behaviour; a request or intention directed at a person. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- head_tremor (score 0.46, threshold 0.30; bouts [[0.0, 3.5]])
- playing_object (score 0.39, threshold 0.34; bouts [[3.5, 4.0]])

**What you reported:**
- no behaviour recognized in the question

**Possible explanations for head_tremor** (recording; kinds in order of fit to your description: medical):
- [medical] idiopathic head tremor: episodes of head bobbing in an alert dog that usually stop when the dog is distracted (called, offered food or a toy); considered benign but diagnosed after other causes are excluded (sources: mspca_ihts; matched in your description: young)
- [medical] a focal seizure, the main alternative: it is not stopped by distraction and may come with drooling, unresponsiveness or other signs (sources: mspca_ihts; matched in your description: -)
- [medical] shaking the head to the side because of ear discomfort is a different movement (a vigorous shake, often with ear scratching) (sources: vin_vestibular; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for playing_object** (recording; kinds in order of fit to your description: play_or_normal, request):
- [play_or_normal] predatory or object play; cats need opportunities for play and predatory behaviour (sources: ellis2013; matched in your description: ball)
- [request] an invitation for interactive play when the animal brings the toy to a person or looks back at them (sources: ellis2013; matched in your description: -)
- request or intention possible: yes

**Context observed in the recording (automatic, can be wrong):**
- a person in view (detector, score 0.54; [[2.0, 2.2]] s)

**Before and after:**
- head_tremor 0.0-3.5 s: before - nothing recognized; after - playing_object (3.5-4.0 s)
- playing_object 3.5-4.0 s: before - a person in view (2.0-2.2 s), head_tremor (0.0-3.5 s); after - nothing recognized

**Would help tell them apart:**
- Does the dog stay alert, and does the movement stop when you call it or offer a treat?
- Is it a small rhythmic bobbing (yes-yes, no-no) or a vigorous shake as if shaking off water or an ear problem?
- How long do episodes last, and is the dog normal between them?
- Does the animal bring the toy to a person or look at them between pounces?

**See a veterinarian promptly if:**
- episodes with unresponsiveness, drooling, collapse or loss of bladder control
- a first episode in an older dog

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [mspca_ihts] MSPCA-Angell: Idiopathic head tremor syndrome. (veterinary_hospital) https://www.mspca.org/clinical/idiopathic-head-tremor-syndrome/
- [vin_vestibular] Veterinary Partner (VIN): Vestibular disease in dogs and cats. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&id=4951444
- [ellis2013] Ellis SLH et al. (2013) AAFP and ISFM feline environmental needs guidelines. J Feline Med Surg 15:219-230 (as summarized by Veterinary Partner). (clinical_review) https://doi.org/10.1177/1098612X13477537

### t05 - stock_mixkit_50675 (confuser)

**Species (given):** dog  
**Scene:** On the floor at home with his teddy bear.  
**Question:** My pit bull puppy shakes and bites his teddy bear really hard. Is he being aggressive?

**Summary:** In the recording: batting, pouncing on or chasing a toy or object (0.0-17.5 s). Movement: the animal stays roughly in place while moving. Rhythmic bobbing, nodding or rotating of the head while awake (user report only) is usually discomfort, itch, pain or a neurological cause. No request is implied by it. Batting, pouncing on or chasing a toy or object (recording) has several possible explanations, in order of fit to your description: ordinary play or maintenance behaviour; a request or intention directed at a person. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- playing_object (score 0.96, threshold 0.34; bouts [[0.0, 17.5]])

**What you reported:**
- head_tremor: not recognized in the recording (it may be outside the clip, brief, or missed by the recognizer)

**Possible explanations for head_tremor** (user report only; kinds in order of fit to your description: medical):
- [medical] shaking the head to the side because of ear discomfort is a different movement (a vigorous shake, often with ear scratching) (sources: vin_vestibular; matched in your description: ear; matched in the recording: ear)
- [medical] idiopathic head tremor: episodes of head bobbing in an alert dog that usually stop when the dog is distracted (called, offered food or a toy); considered benign but diagnosed after other causes are excluded (sources: mspca_ihts; matched in your description: -)
- [medical] a focal seizure, the main alternative: it is not stopped by distraction and may come with drooling, unresponsiveness or other signs (sources: mspca_ihts; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for playing_object** (recording; kinds in order of fit to your description: play_or_normal, request):
- [play_or_normal] predatory or object play; cats need opportunities for play and predatory behaviour (sources: ellis2013; matched in your description: -; matched in the recording: toy)
- [request] an invitation for interactive play when the animal brings the toy to a person or looks back at them (sources: ellis2013; matched in your description: -)
- request or intention possible: yes

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.55; [[0.3, 1.5], [9.2, 9.3], [10.5, 15.0]] s)
- a soft toy (detector, score 0.64; [[4.5, 4.7], [5.8, 6.0], [15.2, 17.3]] s)
- place: veterinary clinic (scene prompts (SigLIP), score 0.87)

**Before and after:**
- playing_object 0.0-17.5 s: before - nothing recognized; after - nothing recognized

**Would help tell them apart:**
- Does the dog stay alert, and does the movement stop when you call it or offer a treat?
- Is it a small rhythmic bobbing (yes-yes, no-no) or a vigorous shake as if shaking off water or an ear problem?
- How long do episodes last, and is the dog normal between them?
- Does the animal bring the toy to a person or look at them between pounces?

**See a veterinarian promptly if:**
- episodes with unresponsiveness, drooling, collapse or loss of bladder control
- a first episode in an older dog

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [vin_vestibular] Veterinary Partner (VIN): Vestibular disease in dogs and cats. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&id=4951444
- [mspca_ihts] MSPCA-Angell: Idiopathic head tremor syndrome. (veterinary_hospital) https://www.mspca.org/clinical/idiopathic-head-tremor-syndrome/
- [ellis2013] Ellis SLH et al. (2013) AAFP and ISFM feline environmental needs guidelines. J Feline Med Surg 15:219-230 (as summarized by Veterinary Partner). (clinical_review) https://doi.org/10.1177/1098612X13477537

### t06 - paper_head_tremor_2015 (medical_confuser)

**Species (given):** dog  
**Scene:** Sitting on the sofa next to me in the evening.  
**Question:** My 2-year-old Schnauzer's head starts bobbing round and round while he is wide awake, and it seems to stop when I talk to him. Should I worry?

**Summary:** In the recording: rhythmic bobbing, nodding or rotating of the head while awake (4.0-9.0 s, 12.0-12.5 s, 18.0-19.5 s, 22.5-24.0 s); being touched or stroked by a person (13.5-18.0 s, 20.0-20.5 s, 26.5-27.5 s). Movement: the animal stays roughly in place while moving. Rhythmic bobbing, nodding or rotating of the head while awake (recording and user report) is usually discomfort, itch, pain or a neurological cause. No request is implied by it. Being touched or stroked by a person (recording) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- head_tremor (score 0.55, threshold 0.30; bouts [[4.0, 9.0], [12.0, 12.5], [18.0, 19.5], [22.5, 24.0]])
- being_petted (score 0.54, threshold 0.30; bouts [[13.5, 18.0], [20.0, 20.5], [26.5, 27.5]])

**What you reported:**
- head_tremor: also seen in the recording

**Possible explanations for head_tremor** (recording and user report; kinds in order of fit to your description: medical):
- [medical] shaking the head to the side because of ear discomfort is a different movement (a vigorous shake, often with ear scratching) (sources: vin_vestibular; matched in your description: ear; matched in the recording: ear)
- [medical] idiopathic head tremor: episodes of head bobbing in an alert dog that usually stop when the dog is distracted (called, offered food or a toy); considered benign but diagnosed after other causes are excluded (sources: mspca_ihts; matched in your description: -)
- [medical] a focal seizure, the main alternative: it is not stopped by distraction and may come with drooling, unresponsiveness or other signs (sources: mspca_ihts; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for being_petted** (recording; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.82; [[5.7, 27.3]] s)

**Before and after:**
- head_tremor 4.0-9.0 s: before - nothing recognized; after - being_petted (13.5-18.0 s)
- head_tremor 12.0-12.5 s: before - nothing recognized; after - being_petted (13.5-18.0 s)
- being_petted 13.5-18.0 s: before - head_tremor (12.0-12.5 s), head_tremor (4.0-9.0 s); after - head_tremor (18.0-19.5 s), head_tremor (22.5-24.0 s)
- being_petted 20.0-20.5 s: before - head_tremor (18.0-19.5 s); after - head_tremor (22.5-24.0 s)

**Would help tell them apart:**
- Does the dog stay alert, and does the movement stop when you call it or offer a treat?
- Is it a small rhythmic bobbing (yes-yes, no-no) or a vigorous shake as if shaking off water or an ear problem?
- How long do episodes last, and is the dog normal between them?

**See a veterinarian promptly if:**
- episodes with unresponsiveness, drooling, collapse or loss of bladder control
- a first episode in an older dog

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- The detector's species vote disagrees with the species you gave (dog); your answer is used, and the detector only locates the animal.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- The recording is silent, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [vin_vestibular] Veterinary Partner (VIN): Vestibular disease in dogs and cats. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&id=4951444
- [mspca_ihts] MSPCA-Angell: Idiopathic head tremor syndrome. (veterinary_hospital) https://www.mspca.org/clinical/idiopathic-head-tremor-syndrome/

### t07 - paper_epilepsy_survey_typeD_2025 (medical_confuser)

**Species (given):** dog  
**Scene:** At home in the living room.  
**Question:** My dog sometimes stands completely still in the living room, staring at nothing, and doesn't react when I call him. Is he just thinking?

**Summary:** In the recording: rhythmic bobbing, nodding or rotating of the head while awake (5.0-17.0 s, 23.0-24.0 s). Staring fixedly or snapping at the air as if catching flies (user report only) has several possible explanations, in order of fit to your description: discomfort, itch, pain or a neurological cause; ordinary play or maintenance behaviour; an abnormal repetitive (compulsive) pattern. No request is implied by it. Barking, whining, howling or growling (user report only) has several possible explanations, in order of fit to your description: arousal, frustration or stress; a request or intention directed at a person; ordinary play or maintenance behaviour. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- head_tremor (score 0.66, threshold 0.30; bouts [[5.0, 17.0], [23.0, 24.0]])

**What you reported:**
- staring_snapping: not recognized in the recording (it may be outside the clip, brief, or missed by the recognizer)
- vocalizing: cannot be checked in this recording (no sound)
- resting: not recognized in the recording (it may be outside the clip, brief, or missed by the recognizer)

**Possible explanations for staring_snapping** (user report only; kinds in order of fit to your description: medical, play_or_normal, compulsive):
- [medical] focal (impaired-awareness) seizures: fly-catching, air snapping, staring and lip licking with reduced responsiveness, which can be mistaken for a behavioural problem (sources: frontiers2026_focal; matched in your description: staring, nothing)
- [play_or_normal] snapping at a real insect or reacting to something actually in view (sources: none; matched in your description: -)
- [compulsive] a compulsive behaviour, the main non-medical explanation weighed for such episodes (sources: frontiers2026_focal, tiira2012; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for vocalizing** (user report only; kinds in order of fit to your description: arousal_or_stress, request, play_or_normal):
- [arousal_or_stress] alerting to a disturbance (a stranger, the doorbell) (sources: yin2004; matched in your description: -)
- [request] seeking contact when separated or isolated (sources: yin2004; matched in your description: -)
- [play_or_normal] play barking during games (sources: yin2004; matched in your description: -)
- request or intention possible: yes

**Possible explanations for resting** (user report only; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for head_tremor** (recording; kinds in order of fit to your description: medical):
- [medical] shaking the head to the side because of ear discomfort is a different movement (a vigorous shake, often with ear scratching) (sources: vin_vestibular; matched in your description: -; matched in the recording: ear)
- [medical] idiopathic head tremor: episodes of head bobbing in an alert dog that usually stop when the dog is distracted (called, offered food or a toy); considered benign but diagnosed after other causes are excluded (sources: mspca_ihts; matched in your description: -)
- [medical] a focal seizure, the main alternative: it is not stopped by distraction and may come with drooling, unresponsiveness or other signs (sources: mspca_ihts; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.53; [[24.2, 24.3]] s)
- place: veterinary clinic (scene prompts (SigLIP), score 0.69)

**Before and after:**
- head_tremor 5.0-17.0 s: before - nothing recognized; after - nothing recognized
- head_tremor 23.0-24.0 s: before - nothing recognized; after - a person close to the animal (24.2-24.3 s)

**Would help tell them apart:**
- Does the dog respond to its name or a touch during an episode?
- Is anything actually there (an insect, a shadow, a light)?
- How long do episodes last, and do they look the same each time?
- What happens just before the barking (a sound outside, being left, a game)?
- Does the dog stay alert, and does the movement stop when you call it or offer a treat?
- Is it a small rhythmic bobbing (yes-yes, no-no) or a vigorous shake as if shaking off water or an ear problem?
- How long do episodes last, and is the dog normal between them?

**See a veterinarian promptly if:**
- episodes of unresponsiveness, drooling, collapse or loss of bladder control (possible seizures)
- episodes with unresponsiveness, drooling, collapse or loss of bladder control
- a first episode in an older dog

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- The recording is silent, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [frontiers2026_focal] Behavioral and autonomic manifestations of suspected focal impaired awareness seizures supported by interictal EEG (three dogs). Front Vet Sci (2026). (peer_reviewed_case_report) https://doi.org/10.3389/fvets.2026.1904889
- [tiira2012] Tiira K et al. (2012) Environmental effects on compulsive tail chasing in dogs. PLoS ONE 7(7): e41684. (peer_reviewed_study) https://doi.org/10.1371/journal.pone.0041684
- [yin2004] Yin S, McCowan B (2004) Barking in domestic dogs: context specificity and individual identification. Anim Behav 68:343-355. (peer_reviewed_study) https://doi.org/10.1016/j.anbehav.2003.07.016
- [vin_vestibular] Veterinary Partner (VIN): Vestibular disease in dogs and cats. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&id=4951444
- [mspca_ihts] MSPCA-Angell: Idiopathic head tremor syndrome. (veterinary_hospital) https://www.mspca.org/clinical/idiopathic-head-tremor-syndrome/

### t08 - paper_sleep_movements_2016 (medical_confuser)

**Species (given):** dog  
**Scene:** Asleep on the rug in the living room.  
**Question:** My retriever kicks and jerks his legs a lot while he sleeps on the rug. Is he just dreaming?

**Summary:** In the recording: suddenly going limp, falling over or collapsing (0.0-40.5 s). Lying down, resting or sleeping (user report only) is usually ordinary play or maintenance behaviour. No request is implied by it. Suddenly going limp, falling over or collapsing (recording) is usually discomfort, itch, pain or a neurological cause. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- collapse (score 0.76, threshold 0.30; bouts [[0.0, 40.5]])

**What you reported:**
- resting: not recognized in the recording (it may be outside the clip, brief, or missed by the recognizer)

**Possible explanations for resting** (user report only; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for collapse** (recording; kinds in order of fit to your description: medical):
- [medical] cataplexy (narcolepsy): sudden limp collapse triggered by excitement such as eating, playing or greeting, with the dog often still following with its eyes and a quick, complete return to normal (sources: vettimes_sleep, vca_narcolepsy; matched in your description: -)
- [medical] a seizure or a fainting episode (heart), which sleep-related collapse can mimic; stiffness, paddling, drooling or confusion afterwards point away from cataplexy (sources: vettimes_sleep; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- people talking (sound model (YAMNet), score 0.59; [[13.0, 13.9], [13.4, 14.4], [13.9, 14.9]] s)
- TV or music (sound model (YAMNet), score 0.5; [[1.4, 2.4]] s)

**Before and after:**
- collapse 0.0-40.5 s: before - nothing recognized; after - nothing recognized

**Would help tell them apart:**
- What was the dog doing just before (eating, playing, greeting, exercising)?
- Was it limp or stiff, and did it follow you with its eyes?
- How fast was it back to normal?

**See a veterinarian promptly if:**
- every first collapse
- the dog cannot be roused, has trouble breathing or the episode lasts more than a few minutes

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [vettimes_sleep] Vet Times: Sleep disorders - differential diagnoses and treatment. (clinical_review) https://www.vettimes.com/news/vets/small-animal-vets/sleep-disorders-differential-diagnoses-and-treatment
- [vca_narcolepsy] VCA Animal Hospitals: Narcolepsy in dogs. (veterinary_hospital) https://vcahospitals.com/know-your-pet/narcolepsy-in-dogs

### t09 - paper_narcolepsy_2021 (medical_confuser)

**Species (given):** dog  
**Scene:** At home and at the vet clinic.  
**Question:** My young French bulldog suddenly goes limp and lies down when he gets excited, then gets up a moment later. What is happening?

**Summary:** In the recording: suddenly going limp, falling over or collapsing (3.0-7.0 s, 9.5-13.5 s, 29.0-50.5 s, 53.5-54.5 s); rhythmic bobbing, nodding or rotating of the head while awake (5.0-5.5 s, 8.0-9.5 s, 20.0-25.5 s, 57.5-60.0 s). Suddenly going limp, falling over or collapsing (recording and user report) is usually discomfort, itch, pain or a neurological cause. No request is implied by it. Lying down, resting or sleeping (user report only) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- collapse (score 0.58, threshold 0.30; bouts [[3.0, 7.0], [9.5, 13.5], [29.0, 50.5], [53.5, 54.5]])
- head_tremor (score 0.42, threshold 0.30; bouts [[5.0, 5.5], [8.0, 9.5], [20.0, 25.5], [57.5, 60.0]])

**What you reported:**
- collapse: also seen in the recording
- resting: not recognized in the recording (it may be outside the clip, brief, or missed by the recognizer)

**Possible explanations for collapse** (recording and user report; kinds in order of fit to your description: medical):
- [medical] cataplexy (narcolepsy): sudden limp collapse triggered by excitement such as eating, playing or greeting, with the dog often still following with its eyes and a quick, complete return to normal (sources: vettimes_sleep, vca_narcolepsy; matched in your description: excited, limp, gets up)
- [medical] a seizure or a fainting episode (heart), which sleep-related collapse can mimic; stiffness, paddling, drooling or confusion afterwards point away from cataplexy (sources: vettimes_sleep; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for resting** (user report only; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for head_tremor** (recording; kinds in order of fit to your description: medical):
- [medical] idiopathic head tremor: episodes of head bobbing in an alert dog that usually stop when the dog is distracted (called, offered food or a toy); considered benign but diagnosed after other causes are excluded (sources: mspca_ihts; matched in your description: bulldog, young)
- [medical] shaking the head to the side because of ear discomfort is a different movement (a vigorous shake, often with ear scratching) (sources: vin_vestibular; matched in your description: -; matched in the recording: ear)
- [medical] a focal seizure, the main alternative: it is not stopped by distraction and may come with drooling, unresponsiveness or other signs (sources: mspca_ihts; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 1.0; [[13.2, 14.2], [19.0, 20.0], [23.3, 23.5]] s)
- place: veterinary clinic (scene prompts (SigLIP), score 0.99)

**Before and after:**
- collapse 3.0-7.0 s: before - nothing recognized; after - head_tremor (8.0-9.5 s)
- collapse 9.5-13.5 s: before - head_tremor (5.0-5.5 s), head_tremor (8.0-9.5 s); after - a person close to the animal (13.2-14.2 s)
- head_tremor 5.0-5.5 s: before - nothing recognized; after - collapse (9.5-13.5 s)
- head_tremor 8.0-9.5 s: before - collapse (3.0-7.0 s); after - a person close to the animal (13.2-14.2 s), collapse (9.5-13.5 s)

**Would help tell them apart:**
- What was the dog doing just before (eating, playing, greeting, exercising)?
- Was it limp or stiff, and did it follow you with its eyes?
- How fast was it back to normal?
- Does the dog stay alert, and does the movement stop when you call it or offer a treat?
- Is it a small rhythmic bobbing (yes-yes, no-no) or a vigorous shake as if shaking off water or an ear problem?
- How long do episodes last, and is the dog normal between them?

**See a veterinarian promptly if:**
- every first collapse
- the dog cannot be roused, has trouble breathing or the episode lasts more than a few minutes
- episodes with unresponsiveness, drooling, collapse or loss of bladder control
- a first episode in an older dog

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- The recording is silent, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [vettimes_sleep] Vet Times: Sleep disorders - differential diagnoses and treatment. (clinical_review) https://www.vettimes.com/news/vets/small-animal-vets/sleep-disorders-differential-diagnoses-and-treatment
- [vca_narcolepsy] VCA Animal Hospitals: Narcolepsy in dogs. (veterinary_hospital) https://vcahospitals.com/know-your-pet/narcolepsy-in-dogs
- [mspca_ihts] MSPCA-Angell: Idiopathic head tremor syndrome. (veterinary_hospital) https://www.mspca.org/clinical/idiopathic-head-tremor-syndrome/
- [vin_vestibular] Veterinary Partner (VIN): Vestibular disease in dogs and cats. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&id=4951444

### t10 - paper_bruxism_2022 (gap)

**Species (given):** dog  
**Scene:** At the vet clinic, wearing a muzzle.  
**Question:** My dog grinds his teeth while he is awake. Why does he do that?

**Summary:** In the recording: biting, sucking or chewing at its own flank, leg or body (0.0-0.5 s, 11.5-16.0 s, 19.5-21.0 s); batting, pouncing on or chasing a toy or object (9.5-11.5 s, 16.5-19.0 s). Movement: the animal stays roughly in place while moving. Biting, sucking or chewing at its own flank, leg or body (recording) has several possible explanations, in order of fit to your description: discomfort, itch, pain or a neurological cause; an abnormal repetitive (compulsive) pattern. No request is implied by it. Batting, pouncing on or chasing a toy or object (recording) has several possible explanations, in order of fit to your description: ordinary play or maintenance behaviour; a request or intention directed at a person. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- self_biting (score 0.49, threshold 0.30; bouts [[0.0, 0.5], [11.5, 16.0], [19.5, 21.0]])
- playing_object (score 0.43, threshold 0.34; bouts [[9.5, 11.5], [16.5, 19.0]])

**What you reported:**
- no behaviour recognized in the question

**Possible explanations for self_biting** (recording; kinds in order of fit to your description: medical, compulsive):
- [medical] pain, including neuropathic pain after an injury or spinal surgery (in the published Dachshund case biting and licking at the flank followed a hemilaminectomy) (sources: frontiers2023_dachshund; matched in your description: -)
- [medical] itch or skin disease at the place it bites (parasites, allergy) (sources: vin_pa, waisglass2006; matched in your description: -)
- [compulsive] a behavioural cause (a compulsive or attention-related pattern), the alternative the published case weighed against pain (sources: frontiers2023_dachshund, cliniciansbrief_pa; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for playing_object** (recording; kinds in order of fit to your description: play_or_normal, request):
- [play_or_normal] predatory or object play; cats need opportunities for play and predatory behaviour (sources: ellis2013; matched in your description: -)
- [request] an invitation for interactive play when the animal brings the toy to a person or looks back at them (sources: ellis2013; matched in your description: -)
- request or intention possible: yes

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.46; [[10.8, 11.5]] s)
- people talking (sound model (YAMNet), score 0.8; [[12.0, 13.0], [13.9, 14.9], [15.4, 16.3]] s)
- place: veterinary clinic (scene prompts (SigLIP), score 0.95)

**Before and after:**
- self_biting 0.0-0.5 s: before - nothing recognized; after - nothing recognized
- self_biting 11.5-16.0 s: before - a person close to the animal (10.8-11.5 s), playing_object (9.5-11.5 s); after - people talking (15.8-16.8 s), people talking (16.8-17.8 s), people talking (17.3-18.2 s), playing_object (16.5-19.0 s)
- playing_object 9.5-11.5 s: before - nothing recognized; after - people talking (12.0-13.0 s), people talking (13.9-14.9 s), people talking (15.4-16.3 s), people talking (15.8-16.8 s), self_biting (11.5-16.0 s)
- playing_object 16.5-19.0 s: before - a person close to the animal (10.8-11.5 s), people talking (12.0-13.0 s), people talking (13.9-14.9 s), people talking (15.4-16.3 s), people talking (15.8-16.8 s), self_biting (11.5-16.0 s); after - self_biting (19.5-21.0 s)

**Would help tell them apart:**
- Did it start after an injury, surgery or a change at home?
- Is there hair loss, redness or a wound where it bites?
- Does it stop when the animal is distracted, and does it happen when nobody is watching?
- Does the animal bring the toy to a person or look at them between pounces?

**See a veterinarian promptly if:**
- open wounds, bleeding or the animal cannot be distracted from biting itself
- biting at the body after spinal surgery or with weakness of the legs

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [frontiers2023_dachshund] Case report: Neuropathic pain versus undesirable behavior in a Dachshund after hemilaminectomy. Front Vet Sci (2023). (peer_reviewed_case_report) https://doi.org/10.3389/fvets.2023.1223800
- [vin_pa] Veterinary Partner (VIN): Psychogenic alopecia. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&catId=254060&id=11924460
- [waisglass2006] Waisglass SE et al. (2006) Underlying medical conditions in cats with presumptive psychogenic alopecia. J Am Vet Med Assoc 228(11):1705-1709. (peer_reviewed_study) https://avmajournals.avma.org/view/journals/javma/228/11/javma.228.11.1705.xml
- [cliniciansbrief_pa] Moffat K. Psychogenic alopecia in cats. Clinician's Brief (2012). (clinical_review) https://www.cliniciansbrief.com/article/psychogenic-alopecia-cats
- [ellis2013] Ellis SLH et al. (2013) AAFP and ISFM feline environmental needs guidelines. J Feline Med Surg 15:219-230 (as summarized by Veterinary Partner). (clinical_review) https://doi.org/10.1177/1098612X13477537

### t11 - paper_eating_seizure_2020 (gap)

**Species (given):** dog  
**Scene:** At the vet clinic with his food bowl.  
**Question:** A few bites into his meal my terrier stops eating, hunches over and starts trembling. Did he eat too fast?

**Summary:** In the recording: eating food (1.0-8.0 s, 12.5-14.5 s); rhythmic bobbing, nodding or rotating of the head while awake (22.5-23.0 s, 27.5-28.0 s, 31.0-31.5 s, 34.0-36.5 s and 2 more); licking or cleaning its own fur, paws or face (38.5-39.0 s, 46.0-48.0 s, 50.0-52.5 s, 56.0-56.5 s and 1 more). Movement: the animal stays roughly in place while moving. Eating food (recording and user report) is usually ordinary play or maintenance behaviour. No request is implied by it. Rhythmic bobbing, nodding or rotating of the head while awake (recording) is usually discomfort, itch, pain or a neurological cause. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- eating (score 0.57, threshold 0.04; bouts [[1.0, 8.0], [12.5, 14.5]])
- head_tremor (score 0.52, threshold 0.30; bouts [[22.5, 23.0], [27.5, 28.0], [31.0, 31.5], [34.0, 36.5]])
- self_grooming (score 0.34, threshold 0.24; bouts [[38.5, 39.0], [46.0, 48.0], [50.0, 52.5], [56.0, 56.5]])

**What you reported:**
- eating: also seen in the recording

**Possible explanations for eating** (recording and user report; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for head_tremor** (recording; kinds in order of fit to your description: medical):
- [medical] shaking the head to the side because of ear discomfort is a different movement (a vigorous shake, often with ear scratching) (sources: vin_vestibular; matched in your description: -; matched in the recording: ear)
- [medical] idiopathic head tremor: episodes of head bobbing in an alert dog that usually stop when the dog is distracted (called, offered food or a toy); considered benign but diagnosed after other causes are excluded (sources: mspca_ihts; matched in your description: -)
- [medical] a focal seizure, the main alternative: it is not stopped by distraction and may come with drooling, unresponsiveness or other signs (sources: mspca_ihts; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for self_grooming** (recording; kinds in order of fit to your description: play_or_normal, arousal_or_stress, medical):
- [play_or_normal] ordinary coat care; cats spend a substantial part of their waking time grooming (sources: vin_pa; matched in your description: -)
- [arousal_or_stress] displacement grooming in conflict, anxiety or frustration (out of context, after a startle or a dispute) (sources: cliniciansbrief_pa, vin_pa; matched in your description: -)
- [medical] itch: when grooming is excessive or focused (hair loss, sores), a medical cause such as allergy or parasites is much more common than a purely behavioural one (sources: waisglass2006, vin_pa; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.76; [[0.2, 3.5], [8.2, 13.7], [15.7, 17.0]] s)
- place: veterinary clinic (scene prompts (SigLIP), score 1.0)

**Before and after:**
- eating 1.0-8.0 s: before - nothing recognized; after - a person close to the animal (8.2-13.7 s)
- eating 12.5-14.5 s: before - nothing recognized; after - a person close to the animal (15.7-17.0 s), a person close to the animal (19.2-19.8 s)
- head_tremor 22.5-23.0 s: before - a person close to the animal (19.2-19.8 s), a person close to the animal (21.5-22.5 s); after - nothing recognized
- head_tremor 27.5-28.0 s: before - a person close to the animal (21.5-22.5 s); after - nothing recognized
- self_grooming 38.5-39.0 s: before - head_tremor (34.0-36.5 s); after - a person close to the animal (40.5-59.5 s), head_tremor (39.5-49.5 s)
- self_grooming 46.0-48.0 s: before - nothing recognized; after - head_tremor (52.5-58.5 s)

**Would help tell them apart:**
- Does the dog stay alert, and does the movement stop when you call it or offer a treat?
- Is it a small rhythmic bobbing (yes-yes, no-no) or a vigorous shake as if shaking off water or an ear problem?
- How long do episodes last, and is the dog normal between them?
- Is there hair loss, short broken hairs or sores where it licks?
- Has the amount of grooming changed recently?
- Does grooming follow stressful events, or happen out of context?

**See a veterinarian promptly if:**
- episodes with unresponsiveness, drooling, collapse or loss of bladder control
- a first episode in an older dog
- open sores or rapidly spreading hair loss
- licking that cannot be interrupted

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- The recording is silent, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [vin_vestibular] Veterinary Partner (VIN): Vestibular disease in dogs and cats. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&id=4951444
- [mspca_ihts] MSPCA-Angell: Idiopathic head tremor syndrome. (veterinary_hospital) https://www.mspca.org/clinical/idiopathic-head-tremor-syndrome/
- [vin_pa] Veterinary Partner (VIN): Psychogenic alopecia. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&catId=254060&id=11924460
- [cliniciansbrief_pa] Moffat K. Psychogenic alopecia in cats. Clinician's Brief (2012). (clinical_review) https://www.cliniciansbrief.com/article/psychogenic-alopecia-cats
- [waisglass2006] Waisglass SE et al. (2006) Underlying medical conditions in cats with presumptive psychogenic alopecia. J Am Vet Med Assoc 228(11):1705-1709. (peer_reviewed_study) https://avmajournals.avma.org/view/journals/javma/228/11/javma.228.11.1705.xml

### t12 - paper_catnip_ethogram_2022 (other)

**Species (given):** cat  
**Scene:** On the floor with a catnip toy.  
**Question:** When I give my cat catnip she bites it and then rubs her head on the floor. Is that normal?

**Summary:** In the recording: rolling on the back or rubbing the face or body on objects (4.5-5.5 s, 10.5-11.0 s, 13.0-13.5 s, 17.0-19.0 s). Rolling on the back or rubbing the face or body on objects (recording and user report) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- rolling_rubbing (score 0.36, threshold 0.30; bouts [[4.5, 5.5], [10.5, 11.0], [13.0, 13.5], [17.0, 19.0]])

**What you reported:**
- rolling_rubbing: also seen in the recording

**Possible explanations for rolling_rubbing** (recording and user report; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] a response to catnip or other cat-attracting plants (silver vine, honeysuckle, valerian): rolling, rubbing, licking and chewing; most cats respond to at least one of these plants (sources: bol2017; matched in your description: catnip, toy)
- [play_or_normal] ordinary scent marking or greeting by rubbing the face on objects or people (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a person in view (detector, score 0.91; [[39.3, 60.0]] s)

**Before and after:**
- rolling_rubbing 4.5-5.5 s: before - nothing recognized; after - nothing recognized
- rolling_rubbing 10.5-11.0 s: before - nothing recognized; after - nothing recognized

**Would help tell them apart:**
- Was a plant, a catnip toy or a new object nearby?
- Does the cat also lick or chew the object?

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [bol2017] Bol S et al. (2017) Responsiveness of cats to silver vine, Tatarian honeysuckle, valerian and catnip. BMC Vet Res 13:70. (peer_reviewed_study) https://doi.org/10.1186/s12917-017-0987-6

### o01 - stock_mixkit_45347 (other)

**Species (given):** cat  
**Scene:** Sitting in the garden.  
**Question:** My cat sits in the garden licking her paw and wiping her face with it. Why does she do that?

**Summary:** In the recording: scratching its own body with a hind foot (5.5-12.0 s); licking or cleaning its own fur, paws or face (5.0-7.5 s, 11.0-11.5 s, 22.0-22.5 s). Movement: the animal stays roughly in place while moving. Licking or cleaning its own fur, paws or face (recording and user report) has several possible explanations, in order of fit to your description: ordinary play or maintenance behaviour; arousal, frustration or stress; discomfort, itch, pain or a neurological cause. No request is implied by it. Rolling on the back or rubbing the face or body on objects (user report only) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- scratching_self (score 0.70, threshold 0.30; bouts [[5.5, 12.0]])
- self_grooming (score 0.40, threshold 0.24; bouts [[5.0, 7.5], [11.0, 11.5], [22.0, 22.5]])

**What you reported:**
- self_grooming: also seen in the recording
- rolling_rubbing: not recognized in the recording (it may be outside the clip, brief, or missed by the recognizer)

**Possible explanations for self_grooming** (recording and user report; kinds in order of fit to your description: play_or_normal, arousal_or_stress, medical):
- [play_or_normal] ordinary coat care; cats spend a substantial part of their waking time grooming (sources: vin_pa; matched in your description: -)
- [arousal_or_stress] displacement grooming in conflict, anxiety or frustration (out of context, after a startle or a dispute) (sources: cliniciansbrief_pa, vin_pa; matched in your description: -)
- [medical] itch: when grooming is excessive or focused (hair loss, sores), a medical cause such as allergy or parasites is much more common than a purely behavioural one (sources: waisglass2006, vin_pa; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for rolling_rubbing** (user report only; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary scent marking or greeting by rubbing the face on objects or people (sources: none; matched in your description: -; matched in the recording: door)
- [play_or_normal] a response to catnip or other cat-attracting plants (silver vine, honeysuckle, valerian): rolling, rubbing, licking and chewing; most cats respond to at least one of these plants (sources: bol2017; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for scratching_self** (recording; kinds in order of fit to your description: play_or_normal, medical):
- [play_or_normal] occasional scratching is ordinary coat care (sources: vin_pa; matched in your description: -)
- [medical] itch from parasites, allergy or ear disease when it is frequent or focused (sources: vin_pa, waisglass2006; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- place: garden or yard (scene prompts (SigLIP), score 0.61)

**Before and after:**
- scratching_self 5.5-12.0 s: before - nothing recognized; after - nothing recognized
- self_grooming 5.0-7.5 s: before - nothing recognized; after - nothing recognized
- self_grooming 11.0-11.5 s: before - nothing recognized; after - nothing recognized

**Would help tell them apart:**
- Is there hair loss, short broken hairs or sores where it licks?
- Has the amount of grooming changed recently?
- Does grooming follow stressful events, or happen out of context?
- Was a plant, a catnip toy or a new object nearby?
- Does the cat also lick or chew the object?
- Is it frequent or focused on one place (ears, neck)?
- Are there fleas, flea dirt, redness or discharge from the ears?

**See a veterinarian promptly if:**
- open sores or rapidly spreading hair loss
- licking that cannot be interrupted
- bleeding or open sores from scratching

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [vin_pa] Veterinary Partner (VIN): Psychogenic alopecia. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&catId=254060&id=11924460
- [cliniciansbrief_pa] Moffat K. Psychogenic alopecia in cats. Clinician's Brief (2012). (clinical_review) https://www.cliniciansbrief.com/article/psychogenic-alopecia-cats
- [waisglass2006] Waisglass SE et al. (2006) Underlying medical conditions in cats with presumptive psychogenic alopecia. J Am Vet Med Assoc 228(11):1705-1709. (peer_reviewed_study) https://avmajournals.avma.org/view/journals/javma/228/11/javma.228.11.1705.xml
- [bol2017] Bol S et al. (2017) Responsiveness of cats to silver vine, Tatarian honeysuckle, valerian and catnip. BMC Vet Res 13:70. (peer_reviewed_study) https://doi.org/10.1186/s12917-017-0987-6

### o02 - stock_mixkit_50689 (other)

**Species (given):** dog  
**Scene:** Lying on the floor with his ball.  
**Question:** My border collie lies on the floor chewing and pushing his ball around for ages. Does he want something from me?

**Summary:** In the recording: batting, pouncing on or chasing a toy or object (0.0-11.0 s). Movement: the animal stays roughly in place while moving. Batting, pouncing on or chasing a toy or object (recording) has several possible explanations, in order of fit to your description: ordinary play or maintenance behaviour; a request or intention directed at a person. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- playing_object (score 0.96, threshold 0.34; bouts [[0.0, 11.0]])

**What you reported:**
- no behaviour recognized in the question

**Possible explanations for playing_object** (recording; kinds in order of fit to your description: play_or_normal, request):
- [play_or_normal] predatory or object play; cats need opportunities for play and predatory behaviour (sources: ellis2013; matched in your description: ball; matched in the recording: toy)
- [request] an invitation for interactive play when the animal brings the toy to a person or looks back at them (sources: ellis2013; matched in your description: -)
- request or intention possible: yes

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.45; [[1.2, 1.3], [3.3, 4.8]] s)
- a frisbee or toy (detector, score 0.58; [[1.2, 1.7], [4.7, 5.0], [6.2, 7.0]] s)
- place: veterinary clinic (scene prompts (SigLIP), score 0.85)

**Before and after:**
- playing_object 0.0-11.0 s: before - nothing recognized; after - nothing recognized

**Would help tell them apart:**
- Does the animal bring the toy to a person or look at them between pounces?

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [ellis2013] Ellis SLH et al. (2013) AAFP and ISFM feline environmental needs guidelines. J Feline Med Surg 15:219-230 (as summarized by Veterinary Partner). (clinical_review) https://doi.org/10.1177/1098612X13477537

### o03 - stock_mixkit_1478 (other)

**Species (given):** dog  
**Scene:** Outside, kneeling next to him.  
**Question:** My golden retriever leans into me and pants while I rub his face. Does he like it?

**Summary:** In the recording: being touched or stroked by a person (1.5-15.5 s). Being touched or stroked by a person (recording) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- being_petted (score 0.85, threshold 0.30; bouts [[1.5, 15.5]])

**What you reported:**
- no behaviour recognized in the question

**Possible explanations for being_petted** (recording; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.53; [[1.2, 15.3]] s)
- place: veterinary clinic (scene prompts (SigLIP), score 0.84)

**Before and after:**
- being_petted 1.5-15.5 s: before - nothing recognized; after - nothing recognized

**Would help tell them apart:**

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**

### o04 - stock_mixkit_45842 (other)

**Species (given):** dog  
**Scene:** In the garden.  
**Question:** My pug stands still with his tongue out while I stroke his head. Is he asking for more?

**Summary:** In the recording: being touched or stroked by a person (1.0-10.5 s). Being touched or stroked by a person (recording and user report) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- being_petted (score 0.82, threshold 0.30; bouts [[1.0, 10.5]])

**What you reported:**
- being_petted: also seen in the recording

**Possible explanations for being_petted** (recording and user report; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.58; [[0.7, 10.2]] s)
- place: veterinary clinic (scene prompts (SigLIP), score 0.8)

**Before and after:**
- being_petted 1.0-10.5 s: before - nothing recognized; after - nothing recognized

**Would help tell them apart:**

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**

### o05 - stock_mixkit_1778 (other)

**Species (given):** cat  
**Scene:** At home.  
**Question:** My cat closes her eyes and lifts her chin when I scratch under it. What does that mean?

**Summary:** In the recording: being touched or stroked by a person (0.0-20.0 s). Movement: the animal stays roughly in place while moving. Scratching its own body with a hind foot (user report only) has several possible explanations, in order of fit to your description: discomfort, itch, pain or a neurological cause; ordinary play or maintenance behaviour. No request is implied by it. Rolling on the back or rubbing the face or body on objects (user report only) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- being_petted (score 0.90, threshold 0.30; bouts [[0.0, 20.0]])

**What you reported:**
- scratching_self: not recognized in the recording (it may be outside the clip, brief, or missed by the recognizer)
- rolling_rubbing: not recognized in the recording (it may be outside the clip, brief, or missed by the recognizer)

**Possible explanations for scratching_self** (user report only; kinds in order of fit to your description: medical, play_or_normal):
- [medical] itch from parasites, allergy or ear disease when it is frequent or focused (sources: vin_pa, waisglass2006; matched in your description: -; matched in the recording: ear)
- [play_or_normal] occasional scratching is ordinary coat care (sources: vin_pa; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for rolling_rubbing** (user report only; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] a response to catnip or other cat-attracting plants (silver vine, honeysuckle, valerian): rolling, rubbing, licking and chewing; most cats respond to at least one of these plants (sources: bol2017; matched in your description: -)
- [play_or_normal] ordinary scent marking or greeting by rubbing the face on objects or people (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for being_petted** (recording; kinds in order of fit to your description: request, play_or_normal, medical):
- [request] a wish for the contact to stop when signs such as tail lashing, skin twitching, ears back or a direct stare appear (petting-induced aggression may follow) (sources: cornell_aggression, ahs_petting, merck_cat_behavior; matched in your description: scratch)
- [play_or_normal] the cat accepts or seeks the contact; many cats like being stroked on the head and neck (sources: none; matched in your description: -)
- [medical] pain at the place touched, when the cat reacts at one spot (a cat in pain may act aggressively to avoid touch) (sources: cornell_aggression; matched in your description: -)
- request or intention possible: yes

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.81; [[0.2, 20.0]] s)
- place: veterinary clinic (scene prompts (SigLIP), score 0.95)

**Before and after:**
- being_petted 0.0-20.0 s: before - nothing recognized; after - nothing recognized

**Would help tell them apart:**
- Is it frequent or focused on one place (ears, neck)?
- Are there fleas, flea dirt, redness or discharge from the ears?
- Was a plant, a catnip toy or a new object nearby?
- Does the cat also lick or chew the object?
- Where is the cat touched when it reacts, and what does its tail and ears do just before?
- Does it seek contact again afterwards?

**See a veterinarian promptly if:**
- bleeding or open sores from scratching
- sudden new sensitivity to touch at one place (possible pain)

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [vin_pa] Veterinary Partner (VIN): Psychogenic alopecia. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&catId=254060&id=11924460
- [waisglass2006] Waisglass SE et al. (2006) Underlying medical conditions in cats with presumptive psychogenic alopecia. J Am Vet Med Assoc 228(11):1705-1709. (peer_reviewed_study) https://avmajournals.avma.org/view/journals/javma/228/11/javma.228.11.1705.xml
- [bol2017] Bol S et al. (2017) Responsiveness of cats to silver vine, Tatarian honeysuckle, valerian and catnip. BMC Vet Res 13:70. (peer_reviewed_study) https://doi.org/10.1186/s12917-017-0987-6
- [cornell_aggression] Cornell Feline Health Center: Feline behavior problems: aggression. (veterinary_school) https://www.vet.cornell.edu/departments-centers-and-institutes/cornell-feline-health-center/health-information/feline-health-topics/feline-behavior-problems-aggression
- [ahs_petting] Animal Humane Society: Petting-induced and play aggression in cats. (animal_welfare_organization) https://www.animalhumanesociety.org/resource/petting-induced-and-play-aggression-cats
- [merck_cat_behavior] Kim SA (2025) Behavior problems of cats. Merck Veterinary Manual. (clinical_review) https://www.merckvetmanual.com/behavior/behavior-of-cats/behavior-problems-of-cats

### o06 - stock_mixkit_38824 (other)

**Species (given):** cat  
**Scene:** On my lap on the sofa.  
**Question:** My cat lets me stroke her on my lap and then suddenly turns over and grabs at my hand. Why?

**Summary:** In the recording: being touched or stroked by a person (0.0-8.0 s, 11.5-12.0 s); lying down, resting or sleeping (8.5-11.0 s, 15.0-15.5 s, 20.0-20.5 s). Being touched or stroked by a person (recording and user report) has several possible explanations, in order of fit to your description: discomfort, itch, pain or a neurological cause; ordinary play or maintenance behaviour; a request or intention directed at a person. Lying down, resting or sleeping (recording) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- being_petted (score 0.78, threshold 0.30; bouts [[0.0, 8.0], [11.5, 12.0]])
- resting (score 0.43, threshold 0.24; bouts [[8.5, 11.0], [15.0, 15.5], [20.0, 20.5]])

**What you reported:**
- being_petted: also seen in the recording

**Possible explanations for being_petted** (recording and user report; kinds in order of fit to your description: medical, play_or_normal, request):
- [medical] pain at the place touched, when the cat reacts at one spot (a cat in pain may act aggressively to avoid touch) (sources: cornell_aggression; matched in your description: suddenly)
- [play_or_normal] the cat accepts or seeks the contact; many cats like being stroked on the head and neck (sources: none; matched in your description: -)
- [request] a wish for the contact to stop when signs such as tail lashing, skin twitching, ears back or a direct stare appear (petting-induced aggression may follow) (sources: cornell_aggression, ahs_petting, merck_cat_behavior; matched in your description: -)
- request or intention possible: yes

**Possible explanations for resting** (recording; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.69; [[0.0, 20.5]] s)
- place: veterinary clinic (scene prompts (SigLIP), score 0.73)

**Before and after:**
- being_petted 0.0-8.0 s: before - nothing recognized; after - resting (8.5-11.0 s)
- being_petted 11.5-12.0 s: before - resting (8.5-11.0 s); after - resting (15.0-15.5 s)
- resting 8.5-11.0 s: before - being_petted (0.0-8.0 s); after - being_petted (11.5-12.0 s)
- resting 15.0-15.5 s: before - being_petted (11.5-12.0 s); after - nothing recognized

**Would help tell them apart:**
- Where is the cat touched when it reacts, and what does its tail and ears do just before?
- Does it seek contact again afterwards?

**See a veterinarian promptly if:**
- sudden new sensitivity to touch at one place (possible pain)

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [cornell_aggression] Cornell Feline Health Center: Feline behavior problems: aggression. (veterinary_school) https://www.vet.cornell.edu/departments-centers-and-institutes/cornell-feline-health-center/health-information/feline-health-topics/feline-behavior-problems-aggression
- [ahs_petting] Animal Humane Society: Petting-induced and play aggression in cats. (animal_welfare_organization) https://www.animalhumanesociety.org/resource/petting-induced-and-play-aggression-cats
- [merck_cat_behavior] Kim SA (2025) Behavior problems of cats. Merck Veterinary Manual. (clinical_review) https://www.merckvetmanual.com/behavior/behavior-of-cats/behavior-problems-of-cats

### o07 - stock_mixkit_42736 (other)

**Species (given):** dog  
**Scene:** On the lawn.  
**Question:** My dog rolls onto his back on the lawn so I can rub his belly. Is he asking me to?

**Summary:** In the recording: being touched or stroked by a person (0.0-8.5 s, 11.0-13.0 s); biting, sucking or chewing at its own flank, leg or body (3.5-7.0 s); suddenly going limp, falling over or collapsing (2.0-2.5 s, 10.5-12.5 s). Being touched or stroked by a person (recording) is usually ordinary play or maintenance behaviour. No request is implied by it. Biting, sucking or chewing at its own flank, leg or body (recording) has several possible explanations, in order of fit to your description: discomfort, itch, pain or a neurological cause; an abnormal repetitive (compulsive) pattern. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- being_petted (score 0.66, threshold 0.30; bouts [[0.0, 8.5], [11.0, 13.0]])
- self_biting (score 0.44, threshold 0.30; bouts [[3.5, 7.0]])
- collapse (score 0.36, threshold 0.30; bouts [[2.0, 2.5], [10.5, 12.5]])

**What you reported:**
- no behaviour recognized in the question

**Possible explanations for being_petted** (recording; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for self_biting** (recording; kinds in order of fit to your description: medical, compulsive):
- [medical] pain, including neuropathic pain after an injury or spinal surgery (in the published Dachshund case biting and licking at the flank followed a hemilaminectomy) (sources: frontiers2023_dachshund; matched in your description: back)
- [medical] itch or skin disease at the place it bites (parasites, allergy) (sources: vin_pa, waisglass2006; matched in your description: -)
- [compulsive] a behavioural cause (a compulsive or attention-related pattern), the alternative the published case weighed against pain (sources: frontiers2023_dachshund, cliniciansbrief_pa; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.8; [[0.0, 12.7]] s)
- place: garden or yard (scene prompts (SigLIP), score 0.9)

**Before and after:**
- being_petted 0.0-8.5 s: before - nothing recognized; after - collapse (10.5-12.5 s)
- being_petted 11.0-13.0 s: before - self_biting (3.5-7.0 s); after - nothing recognized
- self_biting 3.5-7.0 s: before - collapse (2.0-2.5 s); after - being_petted (11.0-13.0 s), collapse (10.5-12.5 s)
- collapse 2.0-2.5 s: before - nothing recognized; after - self_biting (3.5-7.0 s)
- collapse 10.5-12.5 s: before - being_petted (0.0-8.5 s), self_biting (3.5-7.0 s); after - nothing recognized

**Would help tell them apart:**
- Did it start after an injury, surgery or a change at home?
- Is there hair loss, redness or a wound where it bites?
- Does it stop when the animal is distracted, and does it happen when nobody is watching?

**See a veterinarian promptly if:**
- open wounds, bleeding or the animal cannot be distracted from biting itself
- biting at the body after spinal surgery or with weakness of the legs

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- The detector's species vote disagrees with the species you gave (dog); your answer is used, and the detector only locates the animal.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [frontiers2023_dachshund] Case report: Neuropathic pain versus undesirable behavior in a Dachshund after hemilaminectomy. Front Vet Sci (2023). (peer_reviewed_case_report) https://doi.org/10.3389/fvets.2023.1223800
- [vin_pa] Veterinary Partner (VIN): Psychogenic alopecia. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&catId=254060&id=11924460
- [waisglass2006] Waisglass SE et al. (2006) Underlying medical conditions in cats with presumptive psychogenic alopecia. J Am Vet Med Assoc 228(11):1705-1709. (peer_reviewed_study) https://avmajournals.avma.org/view/journals/javma/228/11/javma.228.11.1705.xml
- [cliniciansbrief_pa] Moffat K. Psychogenic alopecia in cats. Clinician's Brief (2012). (clinical_review) https://www.cliniciansbrief.com/article/psychogenic-alopecia-cats

### o08 - stock_mixkit_50687 (other)

**Species (given):** dog  
**Scene:** At home with his food bowl.  
**Question:** My dog eats with his head deep in the bowl and doesn't look up. Is that normal?

**Summary:** In the recording: eating food (0.0-8.5 s). Movement: the animal stays roughly in place while moving. Eating food (recording and user report) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- eating (score 0.99, threshold 0.04; bouts [[0.0, 8.5]])

**What you reported:**
- eating: also seen in the recording

**Possible explanations for eating** (recording and user report; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a food or water bowl (detector, score 0.56; [[0.5, 5.8], [7.0, 8.0]] s)
- place: veterinary clinic (scene prompts (SigLIP), score 0.93)

**Before and after:**
- eating 0.0-8.5 s: before - nothing recognized; after - nothing recognized

**Would help tell them apart:**

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- The detector's species vote disagrees with the species you gave (dog); your answer is used, and the detector only locates the animal.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**

### o09 - stock_mixkit_1547 (other)

**Species (given):** cat  
**Scene:** At home.  
**Question:** My two cats eat from the same bowl at the same time. Is one of them stressed?

**Summary:** In the recording: rolling on the back or rubbing the face or body on objects (0.0-5.5 s, 9.5-15.5 s); eating food (2.5-3.0 s, 6.0-6.5 s, 11.5-12.0 s). Movement: the animal stays roughly in place while moving. Eating food (recording and user report) is usually ordinary play or maintenance behaviour. No request is implied by it. Rolling on the back or rubbing the face or body on objects (recording) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- rolling_rubbing (score 0.56, threshold 0.30; bouts [[0.0, 5.5], [9.5, 15.5]])
- eating (score 0.32, threshold 0.04; bouts [[2.5, 3.0], [6.0, 6.5], [11.5, 12.0]])

**What you reported:**
- eating: also seen in the recording

**Possible explanations for eating** (recording and user report; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for rolling_rubbing** (recording; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] a response to catnip or other cat-attracting plants (silver vine, honeysuckle, valerian): rolling, rubbing, licking and chewing; most cats respond to at least one of these plants (sources: bol2017; matched in your description: -)
- [play_or_normal] ordinary scent marking or greeting by rubbing the face on objects or people (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- place: veterinary clinic (scene prompts (SigLIP), score 0.58)

**Before and after:**
- rolling_rubbing 0.0-5.5 s: before - nothing recognized; after - eating (6.0-6.5 s)
- rolling_rubbing 9.5-15.5 s: before - eating (6.0-6.5 s); after - nothing recognized
- eating 2.5-3.0 s: before - nothing recognized; after - nothing recognized
- eating 6.0-6.5 s: before - rolling_rubbing (0.0-5.5 s); after - rolling_rubbing (9.5-15.5 s)

**Would help tell them apart:**
- Was a plant, a catnip toy or a new object nearby?
- Does the cat also lick or chew the object?

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [bol2017] Bol S et al. (2017) Responsiveness of cats to silver vine, Tatarian honeysuckle, valerian and catnip. BMC Vet Res 13:70. (peer_reviewed_study) https://doi.org/10.1186/s12917-017-0987-6

### o10 - stock_mixkit_33017 (other)

**Species (given):** dog  
**Scene:** On the kitchen floor.  
**Question:** Our three puppies eat from one bowl at the same time. Is that normal?

**Summary:** In the recording: eating food (2.0-19.5 s). Eating food (recording and user report) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- eating (score 0.93, threshold 0.04; bouts [[2.0, 19.5]])

**What you reported:**
- eating: also seen in the recording

**Possible explanations for eating** (recording and user report; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- place: veterinary clinic (scene prompts (SigLIP), score 0.99)

**Before and after:**
- eating 2.0-19.5 s: before - nothing recognized; after - nothing recognized

**Would help tell them apart:**

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**

### o11 - stock_mixkit_31936 (other)

**Species (given):** cat  
**Scene:** Outside our house.  
**Question:** Stray kittens are eating scraps outside our house. Are they very hungry?

**Summary:** In the recording: eating food (0.0-16.0 s). Eating food (recording and user report) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- eating (score 0.89, threshold 0.04; bouts [[0.0, 16.0]])

**What you reported:**
- eating: also seen in the recording

**Possible explanations for eating** (recording and user report; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- place: veterinary clinic (scene prompts (SigLIP), score 0.95)

**Before and after:**
- eating 0.0-16.0 s: before - nothing recognized; after - nothing recognized

**Would help tell them apart:**

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**

### o12 - stock_mixkit_48988 (other)

**Species (given):** dog  
**Scene:** In the garden.  
**Question:** My puppy chews on the grass in the garden. Why does he eat grass?

**Summary:** In the recording: eating food (0.0-0.5 s). Movement: the animal stays roughly in place while moving. Eating food (recording and user report) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- eating (score 0.36, threshold 0.04; bouts [[0.0, 0.5]])

**What you reported:**
- eating: also seen in the recording

**Possible explanations for eating** (recording and user report; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- place: garden or yard (scene prompts (SigLIP), score 0.59)

**Before and after:**
- eating 0.0-0.5 s: before - nothing recognized; after - nothing recognized

**Would help tell them apart:**

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**

### o13 - stock_mixkit_50677 (other)

**Species (given):** dog  
**Scene:** At home.  
**Question:** My dog walks up to his water bowl and laps for a long time. Is he too thirsty?

**Summary:** In the recording: eating food (0.0-2.5 s, 6.0-8.0 s). Walking or running across the scene (user report only) is usually ordinary play or maintenance behaviour. No request is implied by it. Eating food (recording) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- eating (score 0.85, threshold 0.04; bouts [[0.0, 2.5], [6.0, 8.0]])

**What you reported:**
- locomotion: not recognized in the recording (it may be outside the clip, brief, or missed by the recognizer)

**Possible explanations for locomotion** (user report only; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for eating** (recording; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.49; [[0.7, 1.3], [3.5, 5.2], [8.2, 9.2]] s)
- place: veterinary clinic (scene prompts (SigLIP), score 0.98)

**Before and after:**
- eating 0.0-2.5 s: before - nothing recognized; after - a person close to the animal (3.5-5.2 s)
- eating 6.0-8.0 s: before - a person close to the animal (0.7-1.3 s), a person close to the animal (3.5-5.2 s); after - a person close to the animal (8.2-9.2 s)

**Would help tell them apart:**

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**

### o14 - stock_mixkit_4102 (other)

**Species (given):** cat  
**Scene:** On the sofa in the afternoon.  
**Question:** My Siamese sleeps curled up on the sofa most of the afternoon. Is that okay?

**Summary:** In the recording: lying down, resting or sleeping (0.0-19.0 s). Movement: its appearance repeats about every 2.2 s. Lying down, resting or sleeping (recording and user report) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- resting (score 0.64, threshold 0.24; bouts [[0.0, 19.0]])

**What you reported:**
- resting: also seen in the recording

**Possible explanations for resting** (recording and user report; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- place: living room (scene prompts (SigLIP), score 0.62)

**Before and after:**
- resting 0.0-19.0 s: before - nothing recognized; after - nothing recognized

**Would help tell them apart:**

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**

### o15 - stock_mixkit_32319 (other)

**Species (given):** cat  
**Scene:** On the couch.  
**Question:** My kitten sleeps on her back with her paws up on the couch. What does it mean?

**Summary:** In the recording: lying down, resting or sleeping (0.0-10.0 s). Lying down, resting or sleeping (recording and user report) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- resting (score 0.59, threshold 0.24; bouts [[0.0, 10.0]])

**What you reported:**
- resting: also seen in the recording

**Possible explanations for resting** (recording and user report; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- place: veterinary clinic (scene prompts (SigLIP), score 0.9)

**Before and after:**
- resting 0.0-10.0 s: before - nothing recognized; after - nothing recognized

**Would help tell them apart:**

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**

### o16 - stock_mixkit_16619 (other)

**Species (given):** dog  
**Scene:** In his bed at home.  
**Question:** My puppy lies in his bed and watches me. Does he want to come out?

**Summary:** No specific behaviour was recognized with confidence in the recording. Lying down, resting or sleeping (user report only) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- nothing above threshold

**What you reported:**
- resting: not recognized in the recording (it may be outside the clip, brief, or missed by the recognizer)

**Possible explanations for resting** (user report only; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.51; [[0.0, 1.2], [2.5, 4.8]] s)

**Would help tell them apart:**

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**

### o17 - stock_mixkit_48739 (other)

**Species (given):** dog  
**Scene:** On a rug.  
**Question:** Our puppies sleep piled on top of each other. Is that normal?

**Summary:** In the recording: lying down, resting or sleeping (9.0-10.0 s). Lying down, resting or sleeping (recording and user report) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- resting (score 0.35, threshold 0.24; bouts [[9.0, 10.0]])

**What you reported:**
- resting: also seen in the recording

**Possible explanations for resting** (recording and user report; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- place: veterinary clinic (scene prompts (SigLIP), score 0.98)

**Before and after:**
- resting 9.0-10.0 s: before - nothing recognized; after - nothing recognized

**Would help tell them apart:**

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**

### o18 - stock_mixkit_14184 (other)

**Species (given):** cat  
**Scene:** In the garden among the leaves.  
**Question:** Our kittens tumble and wrestle in the leaves in the garden. Are they fighting?

**Summary:** No specific behaviour was recognized with confidence in the recording. Play fighting or wrestling with another animal (user report only) has several possible explanations, in order of fit to your description: ordinary play or maintenance behaviour; arousal, frustration or stress. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- nothing above threshold

**What you reported:**
- social_play: not recognized in the recording (it may be outside the clip, brief, or missed by the recognizer)

**Possible explanations for social_play** (user report only; kinds in order of fit to your description: play_or_normal, arousal_or_stress):
- [play_or_normal] rough-and-tumble play: wrestling with alternating roles and few vocalizations (sources: gajdos2023; matched in your description: wrestl)
- [arousal_or_stress] an agonistic interaction (fighting): chasing in one direction and vocalizing (hissing, growling, yowling) go with fights (sources: gajdos2023; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.45; [[8.5, 8.8]] s)
- place: garden or yard (scene prompts (SigLIP), score 0.99)

**Would help tell them apart:**
- Do the cats take turns chasing and being on top?
- Are there hisses, growls or yowls, and does one cat try to leave?

**See a veterinarian promptly if:**
- injuries or one cat persistently hiding or avoiding the other

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- The detector's species vote disagrees with the species you gave (cat); your answer is used, and the detector only locates the animal.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [gajdos2023] Gajdos-Kmecova N et al. (2023) An ethological analysis of close-contact inter-cat interactions determining if cats are playing, fighting, or something in between. Sci Rep 13:92. (peer_reviewed_study) https://doi.org/10.1038/s41598-022-26121-1

### o19 - stock_mixkit_1779 (other)

**Species (given):** cat  
**Scene:** On the sofa.  
**Question:** My cat sits up and swats at the string toy I wave. Is she enjoying it?

**Summary:** In the recording: batting, pouncing on or chasing a toy or object (5.0-6.5 s, 8.5-14.5 s). Movement: the animal stays roughly in place while moving. Lying down, resting or sleeping (user report only) is usually ordinary play or maintenance behaviour. No request is implied by it. Batting, pouncing on or chasing a toy or object (recording) has several possible explanations, in order of fit to your description: ordinary play or maintenance behaviour; a request or intention directed at a person. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- playing_object (score 0.90, threshold 0.34; bouts [[5.0, 6.5], [8.5, 14.5]])

**What you reported:**
- resting: not recognized in the recording (it may be outside the clip, brief, or missed by the recognizer)

**Possible explanations for resting** (user report only; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for playing_object** (recording; kinds in order of fit to your description: play_or_normal, request):
- [play_or_normal] predatory or object play; cats need opportunities for play and predatory behaviour (sources: ellis2013; matched in your description: toy, string)
- [request] an invitation for interactive play when the animal brings the toy to a person or looks back at them (sources: ellis2013; matched in your description: -)
- request or intention possible: yes

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.7; [[0.3, 8.8], [13.2, 13.3]] s)
- place: living room (scene prompts (SigLIP), score 0.91)

**Before and after:**
- playing_object 5.0-6.5 s: before - nothing recognized; after - nothing recognized
- playing_object 8.5-14.5 s: before - a person close to the animal (0.3-8.8 s); after - nothing recognized

**Would help tell them apart:**
- Does the animal bring the toy to a person or look at them between pounces?

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [ellis2013] Ellis SLH et al. (2013) AAFP and ISFM feline environmental needs guidelines. J Feline Med Surg 15:219-230 (as summarized by Veterinary Partner). (clinical_review) https://doi.org/10.1177/1098612X13477537

### o20 - stock_mixkit_45843 (other)

**Species (given):** dog  
**Scene:** In the park.  
**Question:** My pug runs across the park toward me. Is he happy?

**Summary:** No specific behaviour was recognized with confidence in the recording. Movement: the animal stays roughly in place while moving. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- nothing above threshold

**What you reported:**
- no behaviour recognized in the question

**Context observed in the recording (automatic, can be wrong):**
- a person in view (detector, score 0.73; [[0.0, 5.0]] s)
- place: garden or yard (scene prompts (SigLIP), score 0.53)

**Would help tell them apart:**

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**

### o21 - stock_mixkit_25225 (other)

**Species (given):** dog  
**Scene:** In the snow.  
**Question:** My corgi runs through the snow straight at the camera. Is he excited?

**Summary:** In the recording: turning in tight circles (after its own tail or not) (0.0-18.0 s); rhythmic bobbing, nodding or rotating of the head while awake (2.5-3.0 s, 5.5-6.0 s, 9.0-9.5 s). Movement: the animal stays roughly in place while moving. Turning in tight circles (after its own tail or not) (recording) has several possible explanations, in order of fit to your description: ordinary play or maintenance behaviour; a request or intention directed at a person; arousal, frustration or stress; an abnormal repetitive (compulsive) pattern; discomfort, itch, pain or a neurological cause. Rhythmic bobbing, nodding or rotating of the head while awake (recording) is usually discomfort, itch, pain or a neurological cause. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- tail_chasing (score 0.67, threshold 0.64; bouts [[0.0, 18.0]])
- head_tremor (score 0.38, threshold 0.30; bouts [[2.5, 3.0], [5.5, 6.0], [9.0, 9.5]])

**What you reported:**
- no behaviour recognized in the question

**Possible explanations for tail_chasing** (recording; kinds in order of fit to your description: play_or_normal, request, arousal_or_stress, compulsive, medical):
- [play_or_normal] self-directed play or excitement, especially in young dogs, when bouts are brief and the dog is easy to distract (sources: burn2011, tiira2012; matched in your description: excited)
- [request] a learned way to get attention: owners often laugh at or encourage tail chasing, which can reinforce it; some dogs start when nobody attends to them (sources: burn2011, tiira2012; matched in your description: -)
- [arousal_or_stress] frustration or arousal in a waiting or conflict situation (for example waiting for food, or when the owner talks to other people) (sources: tiira2012; matched in your description: -)
- [compulsive] canine compulsive disorder: habitual, perseverative chasing with lowered responsiveness, often with other repetitive behaviours (light chasing, freezing); more common in some breeds (bull terriers, German shepherds) (sources: tiira2012, burn2011; matched in your description: -)
- [medical] a physical cause near the tail or a neurological one: tail or anal-sac discomfort, skin itch, pain, or focal seizures; behaviour and medical causes can coexist (sources: vanhaevermaet2026, frontiers2026_focal; matched in your description: -)
- [medical] circling that is not tail chasing: tight circles always in the same direction, often with a head tilt or head turn, loss of balance or episodes that cannot be interrupted, point to a balance (vestibular) or brain problem (sources: merck_neuro_exam, dvm360_vestibular, vin_vestibular; matched in your description: -)
- request or intention possible: yes

**Possible explanations for head_tremor** (recording; kinds in order of fit to your description: medical):
- [medical] shaking the head to the side because of ear discomfort is a different movement (a vigorous shake, often with ear scratching) (sources: vin_vestibular; matched in your description: -; matched in the recording: ear)
- [medical] idiopathic head tremor: episodes of head bobbing in an alert dog that usually stop when the dog is distracted (called, offered food or a toy); considered benign but diagnosed after other causes are excluded (sources: mspca_ihts; matched in your description: -)
- [medical] a focal seizure, the main alternative: it is not stopped by distraction and may come with drooling, unresponsiveness or other signs (sources: mspca_ihts; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.56; [[19.0, 19.2]] s)
- place: veterinary clinic (scene prompts (SigLIP), score 0.53)

**Before and after:**
- tail_chasing 0.0-18.0 s: before - nothing recognized; after - a person close to the animal (19.0-19.2 s)
- head_tremor 2.5-3.0 s: before - nothing recognized; after - nothing recognized
- head_tremor 5.5-6.0 s: before - nothing recognized; after - nothing recognized

**Would help tell them apart:**
- How old was the dog when it started, and how often does it happen (occasionally or daily)?
- Can you interrupt it by calling the dog or offering a toy, and does it respond to you during a bout?
- In which situations does it start (waiting for food, when people talk to each other, when left alone, when excited)?
- Does the dog bite or injure the tail, scoot, or show hair loss or sores near the tail base?
- Are there other repetitive behaviours such as chasing lights or shadows, snapping at the air, freezing or staring?
- Does it always circle in the same direction, with a head tilt or loss of balance, even without looking at its tail?
- Does the dog stay alert, and does the movement stop when you call it or offer a treat?
- Is it a small rhythmic bobbing (yes-yes, no-no) or a vigorous shake as if shaking off water or an ear problem?

**See a veterinarian promptly if:**
- the tail is wounded or bleeding
- bouts cannot be interrupted or interfere with eating or sleeping
- episodes with staring, unresponsiveness or snapping at the air (possible seizures)
- circling with a head tilt, falling or loss of balance
- episodes with unresponsiveness, drooling, collapse or loss of bladder control
- a first episode in an older dog

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [burn2011] Burn CC (2011) A vicious cycle: a cross-sectional study of canine tail-chasing and human responses to it, using a free video-sharing website. PLoS ONE 6(11): e26553. (peer_reviewed_study) https://doi.org/10.1371/journal.pone.0026553
- [tiira2012] Tiira K et al. (2012) Environmental effects on compulsive tail chasing in dogs. PLoS ONE 7(7): e41684. (peer_reviewed_study) https://doi.org/10.1371/journal.pone.0041684
- [vanhaevermaet2026] Van Haevermaet et al. (2026) Tail chasing in a dog: false dichotomy between behavioural and medical problems. Veterinary Record Case Reports. (peer_reviewed_case_report) https://doi.org/10.1002/vrc2.70334
- [frontiers2026_focal] Behavioral and autonomic manifestations of suspected focal impaired awareness seizures supported by interictal EEG (three dogs). Front Vet Sci (2026). (peer_reviewed_case_report) https://doi.org/10.3389/fvets.2026.1904889
- [merck_neuro_exam] Merck Veterinary Manual: The neurologic examination of animals. (clinical_review) https://www.merckvetmanual.com/nervous-system/the-neurologic-examination/the-neurologic-examination-of-animals
- [dvm360_vestibular] dvm360: Vestibular disease in dogs and cats (proceedings). (clinical_review) https://www.dvm360.com/view/vestibular-disease-dogs-and-cats-proceedings
- [vin_vestibular] Veterinary Partner (VIN): Vestibular disease in dogs and cats. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&id=4951444
- [mspca_ihts] MSPCA-Angell: Idiopathic head tremor syndrome. (veterinary_hospital) https://www.mspca.org/clinical/idiopathic-head-tremor-syndrome/

### o22 - stock_mixkit_6975 (other)

**Species (given):** cat  
**Scene:** Under the couch, with a feather toy.  
**Question:** My kitten reaches out from under the couch to grab the feather toy. Is she playing or scared?

**Summary:** In the recording: batting, pouncing on or chasing a toy or object (2.0-3.0 s, 5.0-5.5 s, 9.0-11.5 s, 13.5-19.0 s); walking or running across the scene (0.0-1.0 s, 4.0-6.0 s, 8.0-9.0 s, 15.5-16.0 s and 1 more); pushing the front paws alternately into a soft surface (1.0-3.5 s, 6.0-7.5 s, 17.5-18.0 s). Movement: the animal stays roughly in place while moving. Batting, pouncing on or chasing a toy or object (recording and user report) has several possible explanations, in order of fit to your description: ordinary play or maintenance behaviour; a request or intention directed at a person. Walking or running across the scene (recording) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- playing_object (score 0.98, threshold 0.34; bouts [[2.0, 3.0], [5.0, 5.5], [9.0, 11.5], [13.5, 19.0]])
- locomotion (score 0.71, threshold 0.44; bouts [[0.0, 1.0], [4.0, 6.0], [8.0, 9.0], [15.5, 16.0]])
- kneading (score 0.53, threshold 0.41; bouts [[1.0, 3.5], [6.0, 7.5], [17.5, 18.0]])

**What you reported:**
- playing_object: also seen in the recording

**Possible explanations for playing_object** (recording and user report; kinds in order of fit to your description: play_or_normal, request):
- [play_or_normal] predatory or object play; cats need opportunities for play and predatory behaviour (sources: ellis2013; matched in your description: toy)
- [request] an invitation for interactive play when the animal brings the toy to a person or looks back at them (sources: ellis2013; matched in your description: -)
- request or intention possible: yes

**Possible explanations for locomotion** (recording; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for kneading** (recording; kinds in order of fit to your description: play_or_normal, arousal_or_stress):
- [play_or_normal] comfort or contentment behaviour derived from nursing, often with purring, before resting (sources: petmd_knead; matched in your description: -)
- [arousal_or_stress] self-soothing when stressed or uncomfortable, if it is new or excessive (sources: petmd_knead; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 1.0; [[0.0, 10.5], [13.2, 19.0]] s)
- place: living room (scene prompts (SigLIP), score 0.66)

**Before and after:**
- playing_object 2.0-3.0 s: before - locomotion (0.0-1.0 s); after - kneading (6.0-7.5 s), locomotion (4.0-6.0 s), locomotion (8.0-9.0 s)
- playing_object 5.0-5.5 s: before - kneading (1.0-3.5 s), locomotion (0.0-1.0 s); after - kneading (6.0-7.5 s), locomotion (8.0-9.0 s)
- locomotion 0.0-1.0 s: before - nothing recognized; after - kneading (1.0-3.5 s), kneading (6.0-7.5 s), playing_object (2.0-3.0 s), playing_object (5.0-5.5 s)
- locomotion 4.0-6.0 s: before - kneading (1.0-3.5 s), playing_object (2.0-3.0 s); after - kneading (6.0-7.5 s), playing_object (9.0-11.5 s)
- kneading 1.0-3.5 s: before - locomotion (0.0-1.0 s); after - locomotion (4.0-6.0 s), locomotion (8.0-9.0 s), playing_object (5.0-5.5 s)
- kneading 6.0-7.5 s: before - locomotion (0.0-1.0 s), locomotion (4.0-6.0 s), playing_object (2.0-3.0 s), playing_object (5.0-5.5 s); after - locomotion (8.0-9.0 s), playing_object (9.0-11.5 s)

**Would help tell them apart:**
- Does the animal bring the toy to a person or look at them between pounces?
- Is the kneading new or more frequent than usual?
- Does the cat purr and relax afterwards?

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [ellis2013] Ellis SLH et al. (2013) AAFP and ISFM feline environmental needs guidelines. J Feline Med Surg 15:219-230 (as summarized by Veterinary Partner). (clinical_review) https://doi.org/10.1177/1098612X13477537
- [petmd_knead] PetMD: Why do cats knead? (veterinarian_advice) https://www.petmd.com/cat/behavior/why-do-cats-knead

### o23 - stock_mixkit_1494 (other)

**Species (given):** dog  
**Scene:** At the river.  
**Question:** My dog splashes into the river after his ball. Is he playing?

**Summary:** No specific behaviour was recognized with confidence in the recording. Movement: the animal stays roughly in place while moving. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- nothing above threshold

**What you reported:**
- no behaviour recognized in the question

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.5; [[7.3, 7.7], [9.2, 9.7]] s)

**Would help tell them apart:**

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- The detector's species vote disagrees with the species you gave (dog); your answer is used, and the detector only locates the animal.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**

### g01 - stock_mixkit_50683 (gap)

**Species (given):** dog  
**Scene:** In the kitchen, holding a treat.  
**Question:** My border collie pants and stares at the treat in my fingers. Is he begging?

**Summary:** In the recording: eating food (6.0-7.5 s); rhythmic bobbing, nodding or rotating of the head while awake (8.0-9.0 s); licking or cleaning its own fur, paws or face (2.5-3.0 s, 7.5-8.0 s). Movement: the animal stays roughly in place while moving. Eating food (recording) is usually ordinary play or maintenance behaviour. No request is implied by it. Rhythmic bobbing, nodding or rotating of the head while awake (recording) is usually discomfort, itch, pain or a neurological cause. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- eating (score 0.50, threshold 0.04; bouts [[6.0, 7.5]])
- head_tremor (score 0.41, threshold 0.30; bouts [[8.0, 9.0]])
- self_grooming (score 0.39, threshold 0.24; bouts [[2.5, 3.0], [7.5, 8.0]])
- staring_snapping (score 0.30, threshold 0.30; bouts [[4.5, 5.5]])

**What you reported:**
- no behaviour recognized in the question

**Possible explanations for eating** (recording; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for head_tremor** (recording; kinds in order of fit to your description: medical):
- [medical] idiopathic head tremor: episodes of head bobbing in an alert dog that usually stop when the dog is distracted (called, offered food or a toy); considered benign but diagnosed after other causes are excluded (sources: mspca_ihts; matched in your description: treat)
- [medical] shaking the head to the side because of ear discomfort is a different movement (a vigorous shake, often with ear scratching) (sources: vin_vestibular; matched in your description: -; matched in the recording: ear)
- [medical] a focal seizure, the main alternative: it is not stopped by distraction and may come with drooling, unresponsiveness or other signs (sources: mspca_ihts; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.46; [[6.0, 6.5]] s)
- place: veterinary clinic (scene prompts (SigLIP), score 0.97)

**Before and after:**
- eating 6.0-7.5 s: before - a person close to the animal (6.0-6.5 s), self_grooming (2.5-3.0 s), staring_snapping (4.5-5.5 s); after - head_tremor (8.0-9.0 s), self_grooming (7.5-8.0 s)
- head_tremor 8.0-9.0 s: before - a person close to the animal (6.0-6.5 s), eating (6.0-7.5 s), self_grooming (2.5-3.0 s), self_grooming (7.5-8.0 s), staring_snapping (4.5-5.5 s); after - nothing recognized
- self_grooming 2.5-3.0 s: before - nothing recognized; after - a person close to the animal (6.0-6.5 s), eating (6.0-7.5 s), head_tremor (8.0-9.0 s), staring_snapping (4.5-5.5 s)
- self_grooming 7.5-8.0 s: before - a person close to the animal (6.0-6.5 s), eating (6.0-7.5 s), staring_snapping (4.5-5.5 s); after - head_tremor (8.0-9.0 s)

**Would help tell them apart:**
- Does the dog stay alert, and does the movement stop when you call it or offer a treat?
- Is it a small rhythmic bobbing (yes-yes, no-no) or a vigorous shake as if shaking off water or an ear problem?
- How long do episodes last, and is the dog normal between them?

**See a veterinarian promptly if:**
- episodes with unresponsiveness, drooling, collapse or loss of bladder control
- a first episode in an older dog

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [mspca_ihts] MSPCA-Angell: Idiopathic head tremor syndrome. (veterinary_hospital) https://www.mspca.org/clinical/idiopathic-head-tremor-syndrome/
- [vin_vestibular] Veterinary Partner (VIN): Vestibular disease in dogs and cats. (veterinarian_advice) https://veterinarypartner.vin.com/default.aspx?pid=19239&id=4951444

### g02 - stock_mixkit_50686 (gap)

**Species (given):** dog  
**Scene:** At home.  
**Question:** My border collie puts his paw in my hand whenever I hold it out. Does he want a treat?

**Summary:** In the recording: staring fixedly or snapping at the air as if catching flies (2.5-4.0 s); being touched or stroked by a person (3.0-3.5 s, 6.5-7.0 s); licking or cleaning its own fur, paws or face (7.0-8.0 s, 11.0-11.5 s). Movement: the animal stays roughly in place while moving. Staring fixedly or snapping at the air as if catching flies (recording) has several possible explanations, in order of fit to your description: ordinary play or maintenance behaviour; discomfort, itch, pain or a neurological cause; an abnormal repetitive (compulsive) pattern. No request is implied by it. Being touched or stroked by a person (recording) is usually ordinary play or maintenance behaviour. No request is implied by it. The cause is not established from the recording; the questions listed would help tell the candidates apart.

**Seen in the recording:**
- staring_snapping (score 0.40, threshold 0.30; bouts [[2.5, 4.0]])
- being_petted (score 0.34, threshold 0.30; bouts [[3.0, 3.5], [6.5, 7.0]])
- self_grooming (score 0.33, threshold 0.24; bouts [[7.0, 8.0], [11.0, 11.5]])

**What you reported:**
- no behaviour recognized in the question

**Possible explanations for staring_snapping** (recording; kinds in order of fit to your description: play_or_normal, medical, compulsive):
- [play_or_normal] snapping at a real insect or reacting to something actually in view (sources: none; matched in your description: -)
- [medical] focal (impaired-awareness) seizures: fly-catching, air snapping, staring and lip licking with reduced responsiveness, which can be mistaken for a behavioural problem (sources: frontiers2026_focal; matched in your description: -)
- [compulsive] a compulsive behaviour, the main non-medical explanation weighed for such episodes (sources: frontiers2026_focal, tiira2012; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Possible explanations for being_petted** (recording; kinds in order of fit to your description: play_or_normal):
- [play_or_normal] ordinary activity (sources: none; matched in your description: -)
- request or intention possible: no - No request or intention is implied by this behaviour itself.

**Context observed in the recording (automatic, can be wrong):**
- a person close to the animal (detector (person box overlapping the animal), score 0.45; [[3.3, 3.5]] s)
- place: veterinary clinic (scene prompts (SigLIP), score 0.97)

**Before and after:**
- staring_snapping 2.5-4.0 s: before - nothing recognized; after - being_petted (6.5-7.0 s), self_grooming (7.0-8.0 s)
- being_petted 3.0-3.5 s: before - a person close to the animal (3.3-3.5 s); after - self_grooming (7.0-8.0 s)
- being_petted 6.5-7.0 s: before - a person close to the animal (3.3-3.5 s), staring_snapping (2.5-4.0 s); after - self_grooming (11.0-11.5 s), self_grooming (7.0-8.0 s)
- self_grooming 7.0-8.0 s: before - a person close to the animal (3.3-3.5 s), being_petted (3.0-3.5 s), being_petted (6.5-7.0 s), staring_snapping (2.5-4.0 s); after - nothing recognized
- self_grooming 11.0-11.5 s: before - being_petted (6.5-7.0 s); after - nothing recognized

**Would help tell them apart:**
- Does the dog respond to its name or a touch during an episode?
- Is anything actually there (an insect, a shadow, a light)?
- How long do episodes last, and do they look the same each time?

**See a veterinarian promptly if:**
- episodes of unresponsiveness, drooling, collapse or loss of bladder control (possible seizures)

**Uncertainty:**
- The recording can show what the animal does and when; it cannot establish why. Candidate explanations come from the cited sources, ordered by how well your description matches them.
- Behaviour recognition uses a general vision-language model (zero-shot); scores are relative, not probabilities, and its measured accuracy is in results/e31.
- No usable sound in the recording, so vocalizations could not be checked.
- Context taken from the recording (people, objects, sounds, place) comes from automatic detectors and can be wrong; it changes the order of the candidates only half as much as your description.

**Sources:**
- [frontiers2026_focal] Behavioral and autonomic manifestations of suspected focal impaired awareness seizures supported by interictal EEG (three dogs). Front Vet Sci (2026). (peer_reviewed_case_report) https://doi.org/10.3389/fvets.2026.1904889
- [tiira2012] Tiira K et al. (2012) Environmental effects on compulsive tail chasing in dogs. PLoS ONE 7(7): e41684. (peer_reviewed_study) https://doi.org/10.1371/journal.pone.0041684
