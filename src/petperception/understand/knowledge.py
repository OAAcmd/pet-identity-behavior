"""Source-grounded explanation knowledge: what an observed behaviour may mean, of which kind, and what would tell candidates apart.

Rules for entries:
* every candidate explanation cites at least one source in ``SOURCES``; the
  source tier is reported with the answer (peer-reviewed study or case
  report, veterinary-school / clinical review, veterinarian-reviewed advice);
* ``kind`` separates requests and intentions from everything else:
  ``request`` (the animal seeks something from a person: food, attention,
  play), ``play_or_normal`` (ordinary activity, no request implied),
  ``arousal_or_stress`` (frustration, conflict, stress, displacement),
  ``compulsive`` (abnormal repetitive behaviour), ``medical`` (itch, pain,
  skin, neurological) - so an answer never turns every behaviour into "it
  wants something";
* ``context_cues`` are words in the user's scene description or question that
  make a candidate more or less relevant; they change the order, never the
  facts;
* ``distinguish`` lists observations that would separate the candidates;
  ``urgent`` lists signs that warrant prompt veterinary advice.
Nothing here is established for an individual animal from video alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

KNOWLEDGE_VERSION = "behavior-kb-v2"

SOURCES: dict[str, dict] = {
    "tiira2012": {"tier": "peer_reviewed_study", "citation": "Tiira K et al. (2012) Environmental effects on compulsive tail chasing in dogs. PLoS ONE 7(7): e41684.",
                  "url": "https://doi.org/10.1371/journal.pone.0041684",
                  "supports": "questionnaire study of 368 dogs: early onset (3-6 months), lowered responsiveness during bouts in about half of tail chasers, "
                              "co-occurring compulsions; supplementary videos describe chasing when waiting for food, when the owner talks to others, "
                              "or when nobody pays attention"},
    "burn2011": {"tier": "peer_reviewed_study", "citation": "Burn CC (2011) A vicious cycle: a cross-sectional study of canine tail-chasing and human "
                                                         "responses to it, using a free video-sharing website. PLoS ONE 6(11): e26553.",
                 "url": "https://doi.org/10.1371/journal.pone.0026553",
                 "supports": "400 online videos: about a third of tail-chasing dogs chased habitually or were hard to distract; perseverative dogs "
                             "showed fewer play behaviours; laughter in 55 % and encouragement in 43 % of videos, so owner attention may reinforce it"},
    "vanhaevermaet2026": {"tier": "peer_reviewed_case_report", "citation": "Van Haevermaet et al. (2026) Tail chasing in a dog: false dichotomy between "
                                                                          "behavioural and medical problems. Veterinary Record Case Reports.",
                          "url": "https://doi.org/10.1002/vrc2.70334", "supports": "tail chasing can combine behavioural and medical causes"},
    "frontiers2026_focal": {"tier": "peer_reviewed_case_report", "citation": "Behavioral and autonomic manifestations of suspected focal impaired awareness "
                                                                            "seizures supported by interictal EEG (three dogs). Front Vet Sci (2026).",
                            "url": "https://doi.org/10.3389/fvets.2026.1904889",
                            "supports": "fly-catching, air snapping, staring and lip licking with impaired awareness can be focal seizures mistaken for "
                                        "behavioural disorders"},
    "frontiers2023_dachshund": {"tier": "peer_reviewed_case_report", "citation": "Case report: Neuropathic pain versus undesirable behavior in a Dachshund "
                                                                                "after hemilaminectomy. Front Vet Sci (2023).",
                                "url": "https://doi.org/10.3389/fvets.2023.1223800",
                                "supports": "barking and biting / licking / sucking at the flank after spinal surgery; neuropathic pain was a differential "
                                            "for a behaviour seen only at home"},
    "cornell_fhs": {"tier": "veterinary_school", "citation": "Cornell Feline Health Center: Hyperesthesia syndrome.",
                    "url": "https://www.vet.cornell.edu/departments-centers-and-institutes/cornell-feline-health-center/health-information/"
                           "feline-health-topics/hyperesthesia-syndrome",
                    "supports": "skin sensitivity near the base of the tail; scratching, biting, tail chasing, dilated pupils, rippling skin; underlying "
                                "causes (for example disc disease) must be evaluated; stress adds to the reaction"},
    "amengualbatle2019": {"tier": "peer_reviewed_study", "citation": "Amengual Batle P et al. (2019) Feline hyperaesthesia syndrome with self-trauma to the "
                                                                    "tail: retrospective study of seven cases. J Feline Med Surg 21(2).",
                          "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10814607/",
                          "supports": "tail chasing and self-trauma in young cats; hypersensitivity dermatitis suspected in some; fleas should be "
                                      "eliminated even in indoor cats"},
    "tvp_fhs": {"tier": "clinical_review", "citation": "Today's Veterinary Practice: Answering FAQs about feline hyperesthesia syndrome.",
                "url": "https://todaysveterinarypractice.com/behavior/answering-faqs-about-feline-hyperesthesia-syndrome/",
                "supports": "diagnosis of exclusion; consider skin disease and spinal, tail and anal-sac pain; consider focal seizures when episodes "
                            "are stereotyped, abrupt and not interruptible; many signs also occur in healthy cats"},
    "vrc2026_fhs_cat": {"tier": "peer_reviewed_case_report", "citation": "A presumptive diagnosis of feline hyperesthesia syndrome due to stressful "
                                                                        "human-related condition: case report. Vet Res Commun (2026).",
                        "url": "https://doi.org/10.1007/s11259-026-11255-8",
                        "supports": "tail chasing and lumbar tremors began after a household change (birth of a child); the authors state that a "
                                    "causal relationship cannot be established"},
    "mrfrs_vet_tail": {"tier": "veterinarian_advice", "citation": "Merrimack River Feline Rescue Society, Ask a Vet: Why do cats chase their tails?",
                       "url": "https://mrfrs.org/ask-a-vet-with-dr-sam-why-do-cats-chase-their-tails/",
                       "supports": "tail chasing is common play and hunting practice in kittens; a new behaviour in an adult cat should be checked "
                                   "by a veterinarian"},
    "waisglass2006": {"tier": "peer_reviewed_study", "citation": "Waisglass SE et al. (2006) Underlying medical conditions in cats with presumptive "
                                                               "psychogenic alopecia. J Am Vet Med Assoc 228(11):1705-1709.",
                      "url": "https://avmajournals.avma.org/view/journals/javma/228/11/javma.228.11.1705.xml",
                      "supports": "of 21 overgrooming cats presumed psychogenic, 16 had a medical cause of itch, 3 both, 2 a purely behavioural cause"},
    "cliniciansbrief_pa": {"tier": "clinical_review", "citation": "Moffat K. Psychogenic alopecia in cats. Clinician's Brief (2012).",
                           "url": "https://www.cliniciansbrief.com/article/psychogenic-alopecia-cats",
                           "supports": "grooming is a common displacement behaviour in conflict, anxiety or frustration; rewarding it with attention "
                                       "should be avoided"},
    "vin_pa": {"tier": "veterinarian_advice", "citation": "Veterinary Partner (VIN): Psychogenic alopecia.",
               "url": "https://veterinarypartner.vin.com/default.aspx?pid=19239&catId=254060&id=11924460",
               "supports": "itch is the most common sign of skin disease and leads to scratching, licking and biting; behavioural causes link "
                           "to stress, frustration or conflict; diagnosis of exclusion"},
    "petmd_knead": {"tier": "veterinarian_advice", "citation": "PetMD: Why do cats knead?", "url": "https://www.petmd.com/cat/behavior/why-do-cats-knead",
                    "supports": "kneading is usually associated with contentment, comfort or marking; it can be self-soothing when stressed or in pain; "
                                "new or excessive kneading deserves a veterinary check"},
    "gajdos2023": {"tier": "peer_reviewed_study", "citation": "Gajdos-Kmecova N et al. (2023) An ethological analysis of close-contact inter-cat "
                                                            "interactions determining if cats are playing, fighting, or something in between. "
                                                            "Sci Rep 13:92.",
                   "url": "https://doi.org/10.1038/s41598-022-26121-1",
                   "supports": "105 videos: wrestling went with play, vocalization and chasing with agonistic interactions; an intermediate group "
                               "mixed both"},
    "mccomb2009": {"tier": "peer_reviewed_study", "citation": "McComb K et al. (2009) The cry embedded within the purr. Curr Biol 19(13):R507-R508.",
                   "url": "https://doi.org/10.1016/j.cub.2009.05.033",
                   "supports": "cats soliciting food produce purrs with a cry-like component that people rate as more urgent"},
    "prato2020": {"tier": "peer_reviewed_study", "citation": "Prato-Previde E et al. (2020) What's in a meow? A study on human classification and "
                                                           "interpretation of domestic cat vocalizations. Animals 10(12):2390.",
                  "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7765146/",
                  "supports": "meows are mostly human-directed and vary with context (waiting for food, isolation, brushing), but people decode "
                              "the context of single meows poorly"},
    "yin2004": {"tier": "peer_reviewed_study", "citation": "Yin S, McCowan B (2004) Barking in domestic dogs: context specificity and individual "
                                                         "identification. Anim Behav 68:343-355.", "url": "https://doi.org/10.1016/j.anbehav.2003.07.016",
                "supports": "barks differ acoustically between disturbance, isolation and play contexts"},
    "ellis2013": {"tier": "clinical_review", "citation": "Ellis SLH et al. (2013) AAFP and ISFM feline environmental needs guidelines. J Feline Med Surg "
                                                       "15:219-230 (as summarized by Veterinary Partner).", "url": "https://doi.org/10.1177/1098612X13477537",
                  "supports": "cats need opportunities for play and predatory behaviour and predictable interactions with people"},
    "merck_neuro_exam": {"tier": "clinical_review", "citation": "Merck Veterinary Manual: The neurologic examination of animals.",
                         "url": "https://www.merckvetmanual.com/nervous-system/the-neurologic-examination/the-neurologic-examination-of-animals",
                         "supports": "a head tilt usually indicates vestibular dysfunction; animals with forebrain lesions may turn their heads and "
                                     "circle in one direction"},
    "dvm360_vestibular": {"tier": "clinical_review", "citation": "dvm360: Vestibular disease in dogs and cats (proceedings).",
                          "url": "https://www.dvm360.com/view/vestibular-disease-dogs-and-cats-proceedings",
                          "supports": "vestibular disorders cause head tilt, falling, rolling, leaning, circling and abnormal eye movements; circling "
                                      "also occurs with focal forebrain lesions"},
    "vin_vestibular": {"tier": "veterinarian_advice", "citation": "Veterinary Partner (VIN): Vestibular disease in dogs and cats.",
                       "url": "https://veterinarypartner.vin.com/default.aspx?pid=19239&id=4951444",
                       "supports": "middle- and inner-ear problems are separate from brain problems; localizing the cause decides outlook and "
                                   "treatment"},
    "mspca_ihts": {"tier": "veterinary_hospital", "citation": "MSPCA-Angell: Idiopathic head tremor syndrome.",
                   "url": "https://www.mspca.org/clinical/idiopathic-head-tremor-syndrome/",
                   "supports": "most dogs (up to 87 %) can be distracted out of an episode; the dog stays alert; the primary differential is "
                               "focal seizure activity, which distraction does not stop and which may show autonomic signs"},
    "akc_dream_seizure": {"tier": "veterinarian_advice", "citation": "American Kennel Club (J. Klein, chief veterinarian): Dreaming vs. seizures in dogs.",
                          "url": "https://www.akc.org/expert-advice/health/dreaming-vs-seizures-in-dogs/",
                          "supports": "twitching and small noises during sleep are usually dreaming; a dreaming dog wakes when its name is called, "
                                      "a seizing dog cannot easily be woken and may urinate or be disoriented afterwards"},
    "vca_sleep_twitch": {"tier": "veterinary_hospital", "citation": "VCA Animal Hospitals: What does it mean when dogs twitch in their sleep?",
                         "url": "https://vcahospitals.com/resources/behavior-dog/true-or-false-twitching-feet-while-sleeping-is-a-dream",
                         "supports": "brief, relaxed twitching in sleep is dream-related; stiff, long movements with drooling, confusion or "
                                     "incontinence may be a seizure"},
    "vettimes_sleep": {"tier": "clinical_review", "citation": "Vet Times: Sleep disorders - differential diagnoses and treatment.",
                       "url": "https://www.vettimes.com/news/vets/small-animal-vets/sleep-disorders-differential-diagnoses-and-treatment",
                       "supports": "cataplexy: sudden flaccid collapse induced by excitement (food, greeting, play) with abrupt return to normal; "
                                   "REM sleep behaviour disorder: involuntary, sometimes violent movements in REM sleep that stop when the animal "
                                   "is aroused; both can mimic syncope or seizures"},
    "vca_narcolepsy": {"tier": "veterinary_hospital", "citation": "VCA Animal Hospitals: Narcolepsy in dogs.",
                       "url": "https://vcahospitals.com/know-your-pet/narcolepsy-in-dogs",
                       "supports": "cataplexy episodes last seconds to minutes and often occur when the dog is eating, playing or excited"},
    "bol2017": {"tier": "peer_reviewed_study", "citation": "Bol S et al. (2017) Responsiveness of cats to silver vine, Tatarian honeysuckle, valerian "
                                                        "and catnip. BMC Vet Res 13:70.", "url": "https://doi.org/10.1186/s12917-017-0987-6",
                "supports": "nearly all of 100 domestic cats responded to at least one plant; one in three did not respond to catnip, almost 80 % "
                            "responded to silver vine"},
    "bekoff1995": {"tier": "peer_reviewed_study", "citation": "Bekoff M (1995) Play signals as punctuation: the structure of social play in "
                                                          "canids. Behaviour 132(5-6):419-429.", "url": "https://doi.org/10.1163/156853995X00649",
                   "supports": "play bows occur around play bites and other actions that could be misread, marking them as play"},
    "cornell_aggression": {"tier": "veterinary_school", "citation": "Cornell Feline Health Center: Feline behavior problems: aggression.",
                           "url": "https://www.vet.cornell.edu/departments-centers-and-institutes/cornell-feline-health-center/health-information/"
                                  "feline-health-topics/feline-behavior-problems-aggression",
                           "supports": "petting-induced aggression (over-stimulation, the cat controlling when petting ends) is often preceded by "
                                       "dilated pupils, tail lashing and ears moved back; a relaxed cat holds its tail up with no flicking or "
                                       "twitching; seeing an outdoor cat through a window can trigger redirected aggression; cats in pain may "
                                       "act aggressively to avoid touch"},
    "ahs_petting": {"tier": "animal_welfare_organization", "citation": "Animal Humane Society: Petting-induced and play aggression in cats.",
                    "url": "https://www.animalhumanesociety.org/resource/petting-induced-and-play-aggression-cats",
                    "supports": "tail lashing or thumping, shifting body position, skin twitching and direct stares indicate the cat has had "
                                "enough petting"},
    "merck_cat_behavior": {"tier": "clinical_review", "citation": "Kim SA (2025) Behavior problems of cats. Merck Veterinary Manual.",
                           "url": "https://www.merckvetmanual.com/behavior/behavior-of-cats/behavior-problems-of-cats",
                           "supports": "petting-induced aggression from low tolerance of contact or over-stimulation; arousal can be redirected; "
                                       "some cats use threatening displays to remove a fear-evoking stimulus; medical problems must be ruled out "
                                       "before a problem is called behavioural"},
    "snapcats_aggression": {"tier": "animal_welfare_organization", "citation": "SNAP Cats: Feline aggression.",
                            "url": "https://www.snapcats.org/2018/05/08/feline-aggression/",
                            "supports": "a hunting cat slinks with a lowered head and a twitching tail and lunges when prey is in reach"},
}

KINDS = ("request", "play_or_normal", "arousal_or_stress", "compulsive", "medical")
KIND_TEXT = {"request": "a request or intention directed at a person", "play_or_normal": "ordinary play or maintenance behaviour",
             "arousal_or_stress": "arousal, frustration or stress", "compulsive": "an abnormal repetitive (compulsive) pattern",
             "medical": "discomfort, itch, pain or a neurological cause"}


@dataclass(frozen=True)
class Candidate:
    id: str
    kind: str
    text: str
    sources: tuple[str, ...]
    context_cues: dict[str, float] = field(default_factory=dict)  # cue word -> weight (+ more relevant, - less)


@dataclass(frozen=True)
class Entry:
    behavior: str
    species: tuple[str, ...]
    summary: str
    candidates: tuple[Candidate, ...]
    distinguish: tuple[str, ...]
    urgent: tuple[str, ...]
    temporal_notes: str = ""


ENTRIES: dict[tuple[str, str], Entry] = {}


def _add(e: Entry) -> None:
    for sp in e.species:
        ENTRIES[(e.behavior, sp)] = e


_add(Entry("tail_chasing", ("dog",), "Turning in tight circles after the tail. Common, and usually brief; in some dogs it is habitual and hard to interrupt.",
           (Candidate("dog_tc_play", "play_or_normal", "self-directed play or excitement, especially in young dogs, when bouts are brief and the dog is "
                      "easy to distract", ("burn2011", "tiira2012"), {"puppy": 1.0, "young": 0.8, "playing": 0.6, "excited": 0.6, "once": 0.5,
                                                                      "every day": -0.6, "can't stop": -0.8, "hours": -0.8}),
            Candidate("dog_tc_attention", "request", "a learned way to get attention: owners often laugh at or encourage tail chasing, which can "
                      "reinforce it; some dogs start when nobody attends to them", ("burn2011", "tiira2012"),
                      {"watch": 0.6, "laugh": 0.8, "attention": 0.9, "ignored": 0.8, "when i'm busy": 0.8, "talking": 0.6, "guests": 0.5}),
            Candidate("dog_tc_frustration", "arousal_or_stress", "frustration or arousal in a waiting or conflict situation (for example waiting "
                      "for food, or when the owner talks to other people)", ("tiira2012",),
                      {"waiting": 0.9, "food": 0.7, "before meals": 0.9, "visitors": 0.5, "alone": 0.4, "stress": 0.7}),
            Candidate("dog_tc_compulsive", "compulsive", "canine compulsive disorder: habitual, perseverative chasing with lowered responsiveness, "
                      "often with other repetitive behaviours (light chasing, freezing); more common in some breeds (bull terriers, German "
                      "shepherds)", ("tiira2012", "burn2011"),
                      {"every day": 0.9, "all the time": 0.9, "can't stop": 1.0, "won't stop": 1.0, "hours": 0.8, "bull terrier": 0.8,
                       "german shepherd": 0.6, "injur": 0.6, "trance": 0.8}),
            Candidate("dog_tc_medical", "medical", "a physical cause near the tail or a neurological one: tail or anal-sac discomfort, skin itch, pain, "
                      "or focal seizures; behaviour and medical causes can coexist", ("vanhaevermaet2026", "frontiers2026_focal"),
                      {"bite": 0.7, "biting": 0.7, "licking": 0.5, "wound": 1.0, "bleeding": 1.0, "scooting": 0.8, "older": 0.4, "sudden": 0.6,
                       "staring": 0.6, "new": 0.5}),
            Candidate("dog_tc_neuro_circling", "medical", "circling that is not tail chasing: tight circles always in the same direction, often with a "
                      "head tilt or head turn, loss of balance or episodes that cannot be interrupted, point to a balance (vestibular) or brain "
                      "problem", ("merck_neuro_exam", "dvm360_vestibular", "vin_vestibular"),
                      {"same direction": 1.0, "to the left": 0.8, "to the right": 0.8, "head tilt": 1.2, "tilt": 0.8, "balance": 1.0, "falls": 0.6,
                       "stumbl": 0.8, "can't be distracted": 0.8, "cannot be distracted": 0.8, "puppy": 0.0})),
           ("How old was the dog when it started, and how often does it happen (occasionally or daily)?",
            "Can you interrupt it by calling the dog or offering a toy, and does it respond to you during a bout?",
            "In which situations does it start (waiting for food, when people talk to each other, when left alone, when excited)?",
            "Does the dog bite or injure the tail, scoot, or show hair loss or sores near the tail base?",
            "Are there other repetitive behaviours such as chasing lights or shadows, snapping at the air, freezing or staring?",
            "Does it always circle in the same direction, with a head tilt or loss of balance, even without looking at its tail?"),
           ("the tail is wounded or bleeding", "bouts cannot be interrupted or interfere with eating or sleeping",
            "episodes with staring, unresponsiveness or snapping at the air (possible seizures)",
            "circling with a head tilt, falling or loss of balance"),
           "A bout is repeated whole-body turning in place; the number of turns, how long bouts last and whether they recur matter more than a "
           "single turn."))

_add(Entry("tail_chasing", ("cat",), "Turning after or attacking its own tail. Common play in kittens; new or intense tail chasing in an adult cat has "
           "several medical differentials.",
           (Candidate("cat_tc_play", "play_or_normal", "predatory play with the tail as a moving target, typical of kittens and playful young cats",
                      ("mrfrs_vet_tail", "ellis2013"), {"kitten": 1.2, "young": 0.8, "playing": 0.6, "toy": 0.3, "bored": 0.4, "adult": -0.3,
                                                         "old": -0.6, "senior": -0.6, "sudden": -0.4}),
            Candidate("cat_tc_itch", "medical", "itch or skin disease at the tail base (fleas, allergy, other dermatitis) - one reason a cat turns to "
                      "bite or lick the tail", ("amengualbatle2019", "tvp_fhs", "vin_pa"),
                      {"bit": 0.8, "biting": 0.8, "licking": 0.6, "fleas": 1.2, "scratch": 0.6, "hair loss": 1.0, "fur": 0.4, "outdoor": 0.4}),
            Candidate("cat_tc_pain", "medical", "pain in the tail, the lower back or the anal sacs", ("tvp_fhs", "cornell_fhs"),
                      {"touch": 0.6, "petting": 0.6, "when i pet": 0.8, "back": 0.6, "hiss": 0.5, "cry": 0.5, "older": 0.5, "sudden": 0.6}),
            Candidate("cat_tc_fhs", "medical", "feline hyperaesthesia syndrome: brief episodes with rippling skin over the lower back, dilated pupils, "
                      "tail chasing or biting, sudden running and sometimes vocalizing - a diagnosis made after other causes are excluded",
                      ("cornell_fhs", "amengualbatle2019", "tvp_fhs", "vrc2026_fhs_cat"),
                      {"ripple": 1.2, "rippling": 1.2, "twitch": 1.0, "dilated": 0.8, "runs off": 0.8, "sudden": 0.6, "episode": 0.8, "yowl": 0.6}),
            Candidate("cat_tc_seizure", "medical", "focal seizures, when episodes are abrupt, stereotyped and cannot be interrupted",
                      ("tvp_fhs",), {"can't stop": 0.8, "unresponsive": 1.0, "staring": 0.8, "drool": 0.8, "same way": 0.5}),
            Candidate("cat_tc_stress", "arousal_or_stress", "stress or conflict (a household change, other pets); it can add to the episodes but "
                      "is not shown to cause them", ("cornell_fhs", "vrc2026_fhs_cat"),
                      {"new baby": 1.0, "moved": 0.8, "new cat": 0.8, "new dog": 0.6, "change": 0.6, "stress": 0.8}),
            Candidate("cat_tc_neuro_circling", "medical", "circling that is not tail chasing: tight circles always in the same direction, often with a "
                      "head tilt or loss of balance, point to a balance (vestibular) or brain problem", ("merck_neuro_exam", "dvm360_vestibular",
                                                                                                        "vin_vestibular"),
                      {"same direction": 1.0, "to the left": 0.8, "to the right": 0.8, "head tilt": 1.2, "tilt": 0.8, "balance": 1.0, "falls": 0.6,
                       "stumbl": 0.8})),
           ("How old is the cat, and is the tail chasing new or has it done this since kittenhood?",
            "Does the skin on the lower back ripple or twitch, or do the pupils dilate, during an episode? Does the cat run off afterwards?",
            "Does the cat react when you touch the base of the tail or the lower back?",
            "Is there hair loss, scabs, flea dirt or a wound on the tail or near its base?",
            "Can you interrupt an episode (call, toy), and how long does it last?",
            "Does it always circle in the same direction, with a head tilt or loss of balance?"),
           ("the tail is wounded, bleeding or the cat keeps attacking it", "episodes with unresponsiveness, drooling or loss of bladder control",
            "a sudden new behaviour in an adult or senior cat", "circling with a head tilt, falling or loss of balance"),
           "In kittens bouts are short and playful, interleaved with other play; episodic bouts with skin rippling and running off point elsewhere."))

_add(Entry("tail_directed", ("cat", "dog"), "Licking, biting or chewing the tail without necessarily turning in circles.",
           (Candidate("td_groom", "play_or_normal", "ordinary grooming of the tail as part of a grooming sequence", ("vin_pa",), {"grooming": 0.6, "clean": 0.6}),
            Candidate("td_itch", "medical", "itch or skin disease (parasites, allergy)", ("waisglass2006", "vin_pa", "amengualbatle2019"),
                      {"fleas": 1.2, "hair loss": 1.0, "red": 0.6, "scratch": 0.6, "bit": 0.5}),
            Candidate("td_pain", "medical", "pain or a lesion near the tail or anal sacs; in dogs after surgery, neuropathic pain was a differential",
                      ("tvp_fhs", "frontiers2023_dachshund"), {"surgery": 1.0, "after": 0.3, "touch": 0.6, "wound": 1.0}),
            Candidate("td_compulsive", "compulsive", "a compulsive or displacement pattern when medical causes are excluded",
                      ("cliniciansbrief_pa", "tiira2012"), {"every day": 0.6, "stress": 0.6, "can't stop": 0.8})),
           ("Is there hair loss, redness, a wound or flea dirt at the tail?", "Did it start after an injury, surgery or a change at home?",
            "Does the animal stop when distracted?"),
           ("open wounds or bleeding", "the animal cannot be distracted from damaging the tail")))

_add(Entry("self_grooming", ("cat", "dog"), "Licking or cleaning its own coat, paws or face.",
           (Candidate("sg_normal", "play_or_normal", "ordinary coat care; cats spend a substantial part of their waking time grooming", ("vin_pa",),
                      {"after eating": 0.6, "after": 0.3, "relaxed": 0.5, "sunny": 0.3}),
            Candidate("sg_displacement", "arousal_or_stress", "displacement grooming in conflict, anxiety or frustration (out of context, after a "
                      "startle or a dispute)", ("cliniciansbrief_pa", "vin_pa"), {"after a fight": 0.9, "stress": 0.8, "new": 0.4, "conflict": 0.8,
                                                                                  "visitor": 0.5}),
            Candidate("sg_itch", "medical", "itch: when grooming is excessive or focused (hair loss, sores), a medical cause such as allergy or "
                      "parasites is much more common than a purely behavioural one", ("waisglass2006", "vin_pa"),
                      {"bald": 1.2, "hair loss": 1.2, "excessive": 1.0, "overgroom": 1.0, "all the time": 0.8, "fleas": 1.0, "one spot": 0.8})),
           ("Is there hair loss, short broken hairs or sores where it licks?", "Has the amount of grooming changed recently?",
            "Does grooming follow stressful events, or happen out of context?"),
           ("open sores or rapidly spreading hair loss", "licking that cannot be interrupted")))

_add(Entry("scratching_self", ("cat", "dog"), "Scratching the body or ears with a hind foot.",
           (Candidate("sc_normal", "play_or_normal", "occasional scratching is ordinary coat care", ("vin_pa",), {}),
            Candidate("sc_itch", "medical", "itch from parasites, allergy or ear disease when it is frequent or focused", ("vin_pa", "waisglass2006"),
                      {"ear": 0.8, "fleas": 1.2, "all the time": 0.8, "red": 0.6, "head shaking": 0.8})),
           ("Is it frequent or focused on one place (ears, neck)?", "Are there fleas, flea dirt, redness or discharge from the ears?"),
           ("bleeding or open sores from scratching",)))

_add(Entry("kneading", ("cat",), "Rhythmic alternate pushing of the front paws into a soft surface.",
           (Candidate("kn_comfort", "play_or_normal", "comfort or contentment behaviour derived from nursing, often with purring, before resting",
                      ("petmd_knead",), {"blanket": 0.5, "lap": 0.6, "purr": 0.8, "before sleeping": 0.8, "relaxed": 0.6}),
            Candidate("kn_soothing", "arousal_or_stress", "self-soothing when stressed or uncomfortable, if it is new or excessive", ("petmd_knead",),
                      {"new": 0.6, "excessive": 0.8, "stress": 0.8, "moved": 0.6})),
           ("Is the kneading new or more frequent than usual?", "Does the cat purr and relax afterwards?"), ()))

_add(Entry("playing_object", ("cat", "dog"), "Batting, pouncing on, carrying or chasing a toy or object.",
           (Candidate("po_play", "play_or_normal", "predatory or object play; cats need opportunities for play and predatory behaviour",
                      ("ellis2013",), {"toy": 0.6, "string": 0.6, "ball": 0.6}),
            Candidate("po_invite", "request", "an invitation for interactive play when the animal brings the toy to a person or looks back at them",
                      ("ellis2013",), {"brings": 1.0, "drops": 0.8, "looks at me": 0.8, "with me": 0.6})),
           ("Does the animal bring the toy to a person or look at them between pounces?",), ()))

_add(Entry("social_play", ("cat",), "Close-contact interaction with another cat.",
           (Candidate("sp_play", "play_or_normal", "rough-and-tumble play: wrestling with alternating roles and few vocalizations", ("gajdos2023",),
                      {"wrestl": 0.8, "take turns": 1.0, "quiet": 0.6}),
            Candidate("sp_agonistic", "arousal_or_stress", "an agonistic interaction (fighting): chasing in one direction and vocalizing (hissing, "
                      "growling, yowling) go with fights", ("gajdos2023",), {"hiss": 1.0, "growl": 1.0, "yowl": 0.8, "chasing": 0.6, "hiding": 0.6})),
           ("Do the cats take turns chasing and being on top?", "Are there hisses, growls or yowls, and does one cat try to leave?"),
           ("injuries or one cat persistently hiding or avoiding the other",)))

_add(Entry("social_play", ("dog",), "Chasing, wrestling or mouthing with another dog.",
           (Candidate("sp_dog_play", "play_or_normal", "social play: play bows and role reversals mark rough contact as play", ("bekoff1995",),
                      {"puppies": 0.6, "puppy": 0.6, "ball": 0.4, "bow": 1.0, "take turns": 1.0, "wrestl": 0.6}),),
           ("Do the dogs take turns (chaser and chased, on top and below) and pause with play bows?",
            "Does one dog try to leave or hide while the other keeps going?"),
           ("one dog repeatedly trying to escape, yelping or being injured",)))

_add(Entry("threat_display", ("cat", "dog"), "Hissing, growling, flattened ears, arched back or bared teeth directed at something.",
           (Candidate("td_defensive", "arousal_or_stress", "a defensive warning: the animal feels threatened and wants the trigger to keep away",
                      ("gajdos2023", "merck_cat_behavior"), {"stranger": 0.6, "other cat": 0.6, "dog": 0.4, "vet": 0.6}),
            Candidate("td_pain_threat", "medical", "pain can lower the threshold for warning signals, especially when the animal is touched",
                      ("tvp_fhs", "cornell_fhs"), {"touch": 0.8, "when i pet": 0.8, "pick up": 0.8, "sudden": 0.6})),
           ("What is the animal looking at or reacting to?", "Does it happen only when touched at a particular place?"),
           ("sudden new aggression when touched (possible pain)",)))

_add(Entry("vocalizing", ("cat",), "Meowing, purring loudly or yowling.",
           (Candidate("vc_request", "request", "a request directed at people, typically for food or attention; meows are mostly human-directed, and "
                      "food-soliciting purrs contain a cry-like element", ("prato2020", "mccomb2009"),
                      {"morning": 0.8, "food": 1.0, "bowl": 0.9, "kitchen": 0.7, "door": 0.6, "at me": 0.8}),
            Candidate("vc_context", "arousal_or_stress", "isolation or handling (meows recorded during isolation and brushing differ from food meows, "
                      "but people decode single meows poorly)", ("prato2020",), {"alone": 0.8, "brushing": 0.8, "carrier": 0.8})),
           ("When does it happen: before meals, at a closed door, when alone, when handled?", "What does the cat do when you respond?"),
           ("a sudden change in voice or night-time vocalizing in an older cat",)))

_add(Entry("vocalizing", ("dog",), "Barking, whining or howling.",
           (Candidate("vd_disturbance", "arousal_or_stress", "alerting to a disturbance (a stranger, the doorbell)", ("yin2004",),
                      {"door": 0.8, "doorbell": 1.0, "window": 0.6, "stranger": 0.8}),
            Candidate("vd_isolation", "request", "seeking contact when separated or isolated", ("yin2004",), {"alone": 1.0, "crate": 0.6, "left": 0.6}),
            Candidate("vd_play", "play_or_normal", "play barking during games", ("yin2004",), {"play": 0.9, "ball": 0.6, "toy": 0.6})),
           ("What happens just before the barking (a sound outside, being left, a game)?",), ()))

_add(Entry("tail_flicking", ("cat",), "Flicking, twitching or lashing the tail while the rest of the body may be still.",
           (Candidate("tf_prey", "play_or_normal", "attention to prey or a moving target (for example birds outside a window): a hunting cat "
                      "watches with a twitching tail before it lunges", ("snapcats_aggression",),
                      {"bird": 1.0, "window": 0.6, "watching": 0.8, "prey": 1.0, "toy": 0.5, "mouse": 0.8}),
            Candidate("tf_stop", "request", "a signal that the cat wants contact to stop: during petting or handling, tail lashing or twitching "
                      "often comes before a swat or bite", ("cornell_aggression", "ahs_petting", "merck_cat_behavior"),
                      {"pet": 0.8, "petting": 1.0, "stroking": 1.0, "touch": 0.6, "lap": 0.6, "handled": 0.8, "brushing": 0.6}),
            Candidate("tf_irritation", "arousal_or_stress", "irritation or arousal from something it cannot reach or does not like (an outdoor cat "
                      "seen through a window, being disturbed while resting)", ("cornell_aggression",),
                      {"peeved": 1.0, "annoyed": 1.0, "disturb": 0.8, "trying to sleep": 0.8, "other cat": 0.6, "outside": 0.4}),
            Candidate("tf_fhs", "medical", "with rippling skin over the back and sudden running, part of an episode of feline hyperaesthesia",
                      ("cornell_fhs", "tvp_fhs"), {"ripple": 1.2, "rippling": 1.2, "skin twitch": 1.2, "runs off": 0.8})),
           ("What is the cat looking at or reacting to while the tail moves?", "Does it happen while it is being petted or handled, and does a swat "
            "or bite follow?", "Does the skin on the back ripple at the same time?"),
           ("tail flicking with skin rippling, dilated pupils and sudden running or self-attack (possible hyperaesthesia episode)",),
           "A relaxed cat holds its tail up without flicking; the speed and context of the movement matter."))

_add(Entry("staring_snapping", ("dog",), "Staring fixedly, or snapping at the air as if catching flies.",
           (Candidate("ss_insect", "play_or_normal", "snapping at a real insect or reacting to something actually in view", (),
                      {"fly": 0.4, "insect": 1.0, "bee": 1.0, "summer": 0.4}),
            Candidate("ss_seizure", "medical", "focal (impaired-awareness) seizures: fly-catching, air snapping, staring and lip licking with "
                      "reduced responsiveness, which can be mistaken for a behavioural problem", ("frontiers2026_focal",),
                      {"suddenly": 0.8, "sudden": 0.8, "not notice": 1.0, "doesn't respond": 1.0, "unresponsive": 1.0, "trance": 0.8,
                       "staring": 0.6, "nothing": 0.6, "drool": 0.8, "episode": 0.8}),
            Candidate("ss_compulsive", "compulsive", "a compulsive behaviour, the main non-medical explanation weighed for such episodes",
                      ("frontiers2026_focal", "tiira2012"), {"every day": 0.6, "all the time": 0.6, "shadows": 0.8, "lights": 0.8})),
           ("Does the dog respond to its name or a touch during an episode?", "Is anything actually there (an insect, a shadow, a light)?",
            "How long do episodes last, and do they look the same each time?"),
           ("episodes of unresponsiveness, drooling, collapse or loss of bladder control (possible seizures)",)))

_add(Entry("self_biting", ("cat", "dog"), "Biting, sucking or chewing at its own flank, leg or body.",
           (Candidate("sb_neuropathic", "medical", "pain, including neuropathic pain after an injury or spinal surgery (in the published "
                      "Dachshund case biting and licking at the flank followed a hemilaminectomy)", ("frontiers2023_dachshund",),
                      {"surgery": 1.2, "operation": 1.0, "since": 0.3, "injury": 0.8, "back": 0.5, "touch": 0.5}),
            Candidate("sb_itch", "medical", "itch or skin disease at the place it bites (parasites, allergy)", ("vin_pa", "waisglass2006"),
                      {"fleas": 1.2, "hair loss": 1.0, "red": 0.6, "scratch": 0.6}),
            Candidate("sb_behavioural", "compulsive", "a behavioural cause (a compulsive or attention-related pattern), the alternative the "
                      "published case weighed against pain", ("frontiers2023_dachshund", "cliniciansbrief_pa"),
                      {"every day": 0.6, "when alone": 0.6, "attention": 0.6, "only at home": 0.6, "bad behavior": 0.4, "bad behaviour": 0.4})),
           ("Did it start after an injury, surgery or a change at home?", "Is there hair loss, redness or a wound where it bites?",
            "Does it stop when the animal is distracted, and does it happen when nobody is watching?"),
           ("open wounds, bleeding or the animal cannot be distracted from biting itself",
            "biting at the body after spinal surgery or with weakness of the legs")))

_add(Entry("being_petted", ("cat",), "Being touched or stroked by a person.",
           (Candidate("bp_accepts", "play_or_normal", "the cat accepts or seeks the contact; many cats like being stroked on the head and neck", (), {}),
            Candidate("bp_overstim", "request", "a wish for the contact to stop when signs such as tail lashing, skin twitching, ears back or a "
                      "direct stare appear (petting-induced aggression may follow)", ("cornell_aggression", "ahs_petting", "merck_cat_behavior"),
                      {"belly": 0.8, "bite": 1.0, "bites": 1.0, "swat": 1.0, "scratch": 0.6, "too long": 0.8, "tail": 0.4}),
            Candidate("bp_pain", "medical", "pain at the place touched, when the cat reacts at one spot (a cat in pain may act aggressively to "
                      "avoid touch)", ("cornell_aggression",), {"one spot": 1.0, "back": 0.6, "suddenly": 0.6, "hiss": 0.6})),
           ("Where is the cat touched when it reacts, and what does its tail and ears do just before?", "Does it seek contact again afterwards?"),
           ("sudden new sensitivity to touch at one place (possible pain)",)))

_add(Entry("head_tremor", ("dog",), "Rhythmic bobbing, nodding or rotating of the head while the dog is awake.",
           (Candidate("ht_idiopathic", "medical", "idiopathic head tremor: episodes of head bobbing in an alert dog that usually stop when the dog is "
                      "distracted (called, offered food or a toy); considered benign but diagnosed after other causes are excluded",
                      ("mspca_ihts",), {"stops when": 1.0, "distract": 1.0, "treat": 0.6, "alert": 0.8, "bulldog": 0.8, "boxer": 0.8,
                                         "doberman": 0.8, "labrador": 0.5, "young": 0.3}),
            Candidate("ht_focal_seizure", "medical", "a focal seizure, the main alternative: it is not stopped by distraction and may come with "
                      "drooling, unresponsiveness or other signs", ("mspca_ihts",),
                      {"doesn't respond": 1.0, "not respond": 1.0, "drool": 1.0, "unresponsive": 1.0, "can't stop": 0.6, "afterwards": 0.4}),
            Candidate("ht_ear", "medical", "shaking the head to the side because of ear discomfort is a different movement (a vigorous shake, "
                      "often with ear scratching)", ("vin_vestibular",), {"ear": 1.0, "scratch": 0.6, "shakes its head": 0.6, "smell": 0.4})),
           ("Does the dog stay alert, and does the movement stop when you call it or offer a treat?",
            "Is it a small rhythmic bobbing (yes-yes, no-no) or a vigorous shake as if shaking off water or an ear problem?",
            "How long do episodes last, and is the dog normal between them?"),
           ("episodes with unresponsiveness, drooling, collapse or loss of bladder control", "a first episode in an older dog")))

_add(Entry("sleep_movements", ("cat", "dog"), "Twitching, paddling, whimpering or running movements while asleep.",
           (Candidate("sm_dreaming", "play_or_normal", "ordinary dreaming: brief, relaxed twitches, paddling and small noises in REM sleep; the "
                      "animal wakes when its name is called and is normal straight away", ("akc_dream_seizure", "vca_sleep_twitch"),
                      {"dream": 0.8, "brief": 0.6, "wakes up": 0.8, "puppy": 0.6, "kitten": 0.6, "senior": 0.3}),
            Candidate("sm_seizure", "medical", "a seizure during sleep: stiff or violent movements, lasting longer, the animal cannot easily be "
                      "woken, with drooling, urination or confusion afterwards", ("akc_dream_seizure", "vca_sleep_twitch"),
                      {"stiff": 1.0, "rigid": 1.0, "can't wake": 1.2, "cannot wake": 1.2, "won't wake": 1.2, "drool": 0.8, "pee": 0.8,
                       "urinat": 0.8, "confused": 0.8, "minutes": 0.4}),
            Candidate("sm_rbd", "medical", "a REM sleep behaviour disorder: involuntary, sometimes violent movements during sleep that stop as soon "
                      "as the animal is woken", ("vettimes_sleep",), {"violent": 1.0, "every night": 0.8, "bites in his sleep": 0.8,
                                                                      "attacks": 0.6, "stops when woken": 1.0})),
           ("Does the animal wake easily when you call its name, and is it normal right away?",
            "Are the movements soft and brief, or stiff, violent or long?", "Is there drooling, urination or confusion afterwards?"),
           ("the animal cannot be woken, is rigid or convulsing, or the episode lasts more than a few minutes",)))

_add(Entry("collapse", ("dog",), "Suddenly going limp, falling over or collapsing.",
           (Candidate("co_cataplexy", "medical", "cataplexy (narcolepsy): sudden limp collapse triggered by excitement such as eating, playing or "
                      "greeting, with the dog often still following with its eyes and a quick, complete return to normal",
                      ("vettimes_sleep", "vca_narcolepsy"), {"eating": 0.8, "food": 0.8, "play": 0.8, "excited": 1.0, "greet": 0.8, "limp": 0.8,
                                                             "gets up": 0.6, "puppy": 0.4}),
            Candidate("co_seizure_syncope", "medical", "a seizure or a fainting episode (heart), which sleep-related collapse can mimic; stiffness, "
                      "paddling, drooling or confusion afterwards point away from cataplexy", ("vettimes_sleep",),
                      {"stiff": 1.0, "paddl": 0.8, "drool": 0.8, "confused": 0.8, "exercise": 0.5, "older": 0.4})),
           ("What was the dog doing just before (eating, playing, greeting, exercising)?", "Was it limp or stiff, and did it follow you with its eyes?",
            "How fast was it back to normal?"),
           ("every first collapse", "the dog cannot be roused, has trouble breathing or the episode lasts more than a few minutes")))

_add(Entry("rolling_rubbing", ("cat",), "Rolling on the back or rubbing the face, chin or body on objects.",
           (Candidate("rr_plants", "play_or_normal", "a response to catnip or other cat-attracting plants (silver vine, honeysuckle, valerian): "
                      "rolling, rubbing, licking and chewing; most cats respond to at least one of these plants", ("bol2017",),
                      {"catnip": 1.5, "silver vine": 1.5, "valerian": 1.2, "honeysuckle": 1.2, "plant": 0.8, "toy": 0.4}),
            Candidate("rr_marking", "play_or_normal", "ordinary scent marking or greeting by rubbing the face on objects or people", (),
                      {"door": 0.4, "furniture": 0.4, "legs": 0.6, "greet": 0.6})),
           ("Was a plant, a catnip toy or a new object nearby?", "Does the cat also lick or chew the object?"), ()))

for _b, _summary in (("eating", "Eating."), ("drinking", "Drinking."), ("resting", "Lying down, resting or sleeping."),
                     ("locomotion", "Walking or running."), ("being_petted", "Being touched or stroked by a person.")):
    _add(Entry(_b, ("dog",) if _b == "being_petted" else ("cat", "dog"), _summary + " Ordinary activity; no request or problem is implied by the behaviour itself.",
               (Candidate(f"{_b}_normal", "play_or_normal", "ordinary activity", (), {}),), (), ()))


def entry(behavior: str, species: str) -> Entry | None:
    return ENTRIES.get((behavior, species))


def validate() -> list[str]:
    errs = []
    for (b, sp), e in ENTRIES.items():
        for c in e.candidates:
            if c.kind not in KINDS:
                errs.append(f"{b}/{sp}/{c.id}: kind {c.kind}")
            for s in c.sources:
                if s not in SOURCES:
                    errs.append(f"{b}/{sp}/{c.id}: unknown source {s}")
            if not c.sources and c.kind != "play_or_normal":
                errs.append(f"{b}/{sp}/{c.id}: non-trivial candidate without a source")
    return errs
