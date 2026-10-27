"""
Preventive measures and daily/weekly routines for each of the 41 conditions.
General wellness guidance only, not medical advice.
"""
import re

_DEFAULT = {
    'summary': 'Maintain a healthy lifestyle and see a doctor for a proper evaluation.',
    'measures': ['Eat a balanced diet', 'Sleep 7-8 hours', 'Stay hydrated', 'Avoid smoking and excess alcohol'],
    'daily': ['Morning: 20-30 min walk or light exercise', 'Meals: balanced, home-cooked, regular timings', 'Night: 7-8 hours of sleep'],
    'weekly': ['Review symptoms and book a doctor visit if they persist'],
    'avoid': ['Self-medicating', 'Skipping medical follow-ups'],
}


def _p(summary, measures, daily, weekly, avoid):
    return {'summary': summary, 'measures': measures, 'daily': daily, 'weekly': weekly, 'avoid': avoid}


_DATA = {
 'fungal infection': _p('Keep skin clean and dry.', ['Dry skin fully after bathing','Wear breathable cotton clothes','Do not share towels or shoes','Keep nails short and clean'], ['Morning: bathe, dry skin folds thoroughly','Day: change sweaty clothes/socks','Night: apply doctor-advised antifungal'], ['Wash bedding and towels in hot water','Disinfect footwear'], ['Tight synthetic clothing','Walking barefoot in public wet areas']),
 'allergy': _p('Identify and avoid triggers.', ['Track triggers (dust, pollen, foods)','Keep home dust-free','Use mattress/pillow covers','Keep antihistamines per doctor advice'], ['Morning: check pollen/air quality, shower after outdoors','Day: keep windows closed in high-pollen hours','Night: wash face, change clothes'], ['Wash bedding at 60C','Vacuum with HEPA filter'], ['Known trigger foods','Smoke and strong perfumes']),
 'gerd': _p('Reduce acid reflux with meal and posture habits.', ['Eat smaller meals','Do not lie down 3 hours after eating','Raise head of bed','Maintain healthy weight'], ['Morning: light breakfast, avoid empty stomach for long','Day: 4-5 small meals','Night: dinner 3 hours before sleep'], ['Keep a food diary to find triggers'], ['Spicy/oily food','Caffeine, alcohol, smoking','Tight belts']),
 'chronic cholestasis': _p('Protect liver and bile flow; medical follow-up is essential.', ['Low-fat diet','Avoid alcohol completely','Regular liver function tests','Take prescribed medicines on time'], ['Morning: fruit, light breakfast','Day: low-fat meals, plenty of water','Night: early light dinner'], ['Monitor itching, urine and stool colour'], ['Alcohol','Fatty/fried food','Unprescribed supplements']),
 'drug reaction': _p('Prevent repeat reactions.', ['Keep a list of drugs that caused reactions','Tell every doctor/pharmacist about it','Read labels before taking medicine','Carry an allergy card'], ['Daily: take only prescribed medicines','Check new medicines with a pharmacist'], ['Review medicine list'], ['Self-medicating','Re-using the offending drug']),
 'peptic ulcer diseae': _p('Protect the stomach lining.', ['Avoid NSAIDs (ibuprofen/aspirin) unless advised','Manage stress','Complete H. pylori treatment if prescribed','Stop smoking'], ['Morning: breakfast within an hour of waking','Day: regular small meals','Night: no late heavy meals'], ['Follow-up if pain or black stools'], ['Alcohol','Spicy food','Skipping meals']),
 'aids': _p('Prevent transmission and protect immunity; follow specialist care.', ['Practice safe sex and use condoms','Never share needles','Take ART exactly as prescribed','Regular viral load/CD4 checks'], ['Morning: take medicines at same time daily','Day: nutritious meals, hygiene','Night: adequate sleep'], ['Check medicine stock','Mental health check-in'], ['Missing ART doses','Sharing razors/needles']),
 'diabetes': _p('Control blood sugar through diet, activity and monitoring.', ['Limit sugar and refined carbs','Exercise 150 min/week','Maintain healthy weight','Regular HbA1c and foot checks'], ['Morning: check fasting sugar, 30 min walk','Day: balanced meals every 3-4 hours, high fibre','Night: light dinner, check feet'], ['Weekly weight and sugar log','Monthly doctor review'], ['Sugary drinks','Skipping meals/medicines','Smoking']),
 'gastroenteritis': _p('Prevent infection and dehydration.', ['Wash hands with soap','Drink clean/boiled water','Eat freshly cooked food','Use ORS during diarrhoea'], ['Morning: ORS and fluids','Day: bland food (rice, banana, toast)','Night: rest'], ['Disinfect kitchen and toilet'], ['Street/raw food','Dairy and oily food while ill']),
 'bronchial asthma': _p('Avoid triggers and keep inhaler ready.', ['Carry reliever inhaler','Avoid smoke, dust and cold air','Use preventer inhaler as prescribed','Get flu vaccine'], ['Morning: peak-flow check, preventer inhaler','Day: avoid triggers, breathing exercises','Night: keep room dust-free'], ['Wash bedding hot','Review asthma action plan'], ['Smoking/smoke exposure','Sudden heavy exertion in cold air']),
 'hypertension': _p('Keep blood pressure in a healthy range.', ['Reduce salt below 5 g/day','Exercise regularly','Limit alcohol, quit smoking','Manage stress and weight'], ['Morning: check BP, 30 min brisk walk, take medicine','Day: low-salt meals, fruits/vegetables','Night: 10 min relaxation, 7-8 h sleep'], ['Log BP readings','Doctor review monthly'], ['Pickles, chips, processed food','Skipping BP medicines']),
 'migraine': _p('Identify triggers and keep a steady routine.', ['Sleep at fixed times','Do not skip meals','Stay hydrated','Limit screen glare and caffeine'], ['Morning: water, regular breakfast','Day: screen breaks every 30 min','Night: dark quiet room, fixed bedtime'], ['Keep a headache diary'], ['Known triggers (cheese, alcohol, strong smell)','Overusing painkillers']),
 'cervical spondylosis': _p('Protect the neck and improve posture.', ['Neck-strengthening exercises','Ergonomic desk setup','Use a thin firm pillow','Avoid long phone-down posture'], ['Morning: gentle neck stretches','Day: break every 30-45 min, keep screen at eye level','Night: proper pillow'], ['Physiotherapy session/exercises 3x'], ['Heavy lifting','Sleeping on thick pillows']),
 'paralysis (brain hemorrhage)': _p('Prevent stroke: control BP, sugar and cholesterol. Emergency condition.', ['Control blood pressure strictly','Quit smoking','Manage diabetes and cholesterol','Know FAST stroke signs'], ['Morning: BP check and medicines','Day: low-salt healthy diet, walk','Night: sleep well, avoid stress'], ['Regular lipid and BP checks'], ['Smoking and alcohol','Ignoring headaches with weakness']),
 'jaundice': _p('Protect the liver.', ['Drink boiled/clean water','Avoid alcohol','Get hepatitis A/B vaccines','Rest adequately'], ['Morning: fresh fruit juice, water','Day: light low-fat meals','Night: rest'], ['Liver function test follow-up'], ['Alcohol','Oily food','Unprescribed painkillers']),
 'malaria': _p('Prevent mosquito bites.', ['Use mosquito nets and repellent','Remove stagnant water','Wear full sleeves at dusk','Consider prophylaxis when travelling to endemic areas'], ['Evening: close windows, repellent','Night: sleep under net'], ['Clean coolers, pots and drains'], ['Stagnant water around home','Ignoring fever after travel']),
 'chicken pox': _p('Vaccinate and limit spread.', ['Varicella vaccine','Isolate until blisters crust','Keep nails short','Wash hands often'], ['Daily: cool bath, loose cotton clothes','Keep fluids up'], ['Disinfect shared items'], ['Scratching blisters','Contact with pregnant/immunocompromised people']),
 'dengue': _p('Prevent Aedes mosquito breeding and bites.', ['Empty standing water weekly','Use repellent and nets by day','Wear long sleeves','Seek care for warning signs'], ['Morning/day: repellent, cover skin','Day: drink plenty of fluids'], ['Empty coolers, buckets, plant trays'], ['Aspirin/ibuprofen','Stagnant water']),
 'typhoid': _p('Safe food and water.', ['Typhoid vaccine','Drink boiled/filtered water','Wash hands before meals','Avoid street food'], ['Daily: wash hands, eat home-cooked food','Drink boiled water'], ['Clean water storage'], ['Raw salads/ice from unknown sources','Poor hand hygiene']),
 'hepatitis a': _p('Hygiene and vaccination.', ['Hepatitis A vaccine','Clean drinking water','Hand washing','Avoid raw shellfish'], ['Daily: wash hands, eat fresh food'], ['Disinfect kitchen/toilet'], ['Street food','Contaminated water']),
 'hepatitis b': _p('Vaccination and avoiding blood exposure.', ['Hepatitis B vaccine','Safe sex','No shared needles/razors','Screen before transfusion'], ['Daily: healthy diet, no alcohol'], ['Regular liver checks if carrier'], ['Alcohol','Sharing personal items']),
 'hepatitis c': _p('Avoid blood exposure; treatable with antivirals.', ['No shared needles','Use sterile equipment for tattoos/piercings','Get screened','Complete antiviral treatment'], ['Daily: healthy diet, no alcohol'], ['Liver check-ups'], ['Alcohol','Sharing razors/needles']),
 'hepatitis d': _p('Prevent by preventing hepatitis B.', ['Hepatitis B vaccine','Avoid shared needles','Safe sex','Regular liver monitoring'], ['Daily: no alcohol, balanced diet'], ['Specialist follow-up'], ['Alcohol','Unsafe injections']),
 'hepatitis e': _p('Clean water and food safety.', ['Drink boiled water','Cook meat (esp. pork) well','Hand washing','Extra care in pregnancy'], ['Daily: boiled water, hygiene'], ['Check water source'], ['Undercooked meat','Untreated water']),
 'alcoholic hepatitis': _p('Stop alcohol to protect the liver.', ['Quit alcohol completely','Join support programme','Nutritious diet with protein','Regular liver tests'], ['Morning: balanced breakfast, water','Day: avoid triggers, stay busy','Night: sleep and support contact'], ['Support group/counselling'], ['Any alcohol','Unprescribed paracetamol overuse']),
 'tuberculosis': _p('Complete treatment and prevent spread.', ['BCG vaccination in children','Finish full course of medicines','Cover cough, ventilate rooms','Screen close contacts'], ['Morning: take TB medicines at the same time','Day: good ventilation, nutritious high-protein food','Night: rest'], ['Monthly clinic follow-up'], ['Skipping doses','Smoking and alcohol']),
 'common cold': _p('Hygiene and rest.', ['Wash hands often','Avoid touching face','Sleep well','Vitamin C foods'], ['Morning: warm fluids','Day: steam, hydration','Night: rest, saline rinse'], ['Clean frequently touched surfaces'], ['Close contact with sick people','Antibiotics without advice']),
 'pneumonia': _p('Vaccinate and protect lungs.', ['Pneumococcal and flu vaccines','Quit smoking','Hand hygiene','Treat colds early'], ['Morning: deep-breathing exercises','Day: fluids, nutritious food','Night: rest, head slightly raised'], ['Follow-up if cough/fever persists'], ['Smoking','Ignoring breathlessness']),
 'dimorphic hemmorhoids(piles)': _p('Prevent constipation and straining.', ['High-fibre diet','2-3 L water daily','Do not strain','Do not sit long on toilet'], ['Morning: warm water, fibre breakfast','Day: walk, fluids','Night: sitz bath if advised'], ['Regular exercise'], ['Spicy/low-fibre food','Prolonged sitting']),
 'heart attack': _p('Lower cardiac risk. Chest pain is an emergency: call emergency services.', ['Quit smoking','Control BP, sugar and cholesterol','Exercise 150 min/week','Heart-healthy diet'], ['Morning: BP check, 30 min walk, medicines','Day: low-salt, low-fat meals','Night: stress relief, 7-8 h sleep'], ['Lipid/ECG check-ups as advised'], ['Smoking','Trans fats and fried food','Ignoring chest pain']),
 'varicose veins': _p('Support circulation.', ['Exercise legs regularly','Elevate legs','Compression stockings if advised','Maintain healthy weight'], ['Morning: walk, wear stockings','Day: avoid standing/sitting >1 h','Night: legs raised 15 min'], ['Swimming or cycling'], ['Prolonged standing','Tight clothing']),
 'hypothyroidism': _p('Keep thyroid hormone stable.', ['Take thyroxine on empty stomach','Regular TSH tests','Adequate iodine','Exercise'], ['Morning: medicine 30-60 min before breakfast','Day: balanced meals, activity'], ['Avoid calcium/iron near dose'], ['Skipping tablets','Excess soy/raw cruciferous in large amounts']),
 'hyperthyroidism': _p('Control thyroid overactivity.', ['Take medicines as prescribed','Regular thyroid tests','Avoid excess iodine','Manage stress'], ['Morning: medicine, calm start','Day: regular meals, avoid caffeine','Night: sleep well'], ['Monitor heart rate and weight'], ['Caffeine','Smoking']),
 'hypoglycemia': _p('Prevent sugar drops. Severe lows are an emergency.', ['Eat regularly','Carry fast sugar (glucose/juice)','Monitor sugar','Adjust medicines with doctor'], ['Morning: do not skip breakfast','Day: meal/snack every 3-4 h','Night: bedtime snack if advised'], ['Review sugar log'], ['Skipping meals','Alcohol on empty stomach']),
 'osteoarthristis': _p('Protect joints.', ['Maintain healthy weight','Low-impact exercise','Strengthen muscles','Use supportive footwear'], ['Morning: joint warm-up stretches','Day: walk/swim, avoid overuse','Night: warm compress'], ['Physio exercises 3x'], ['Heavy lifting','Long squatting/stairs']),
 'arthritis': _p('Keep joints mobile.', ['Gentle regular exercise','Anti-inflammatory diet','Healthy weight','Take medicines as prescribed'], ['Morning: warm shower, stretches','Day: movement breaks','Night: warm compress'], ['Rheumatology follow-up'], ['Smoking','Prolonged inactivity']),
 '(vertigo) paroymsal  positional vertigo': _p('Avoid sudden head movements and falls.', ['Rise slowly','Sleep with head slightly raised','Do Epley/Brandt-Daroff exercises as taught','Fall-proof home'], ['Morning: sit a minute before standing','Day: avoid abrupt head turns','Night: good lighting'], ['Repeat positional exercises'], ['Sudden head movement','Driving during an episode']),
 'acne': _p('Gentle skin care.', ['Wash face twice daily','Use non-comedogenic products','Do not pick pimples','Low-glycaemic diet'], ['Morning: cleanse, moisturise, sunscreen','Night: remove makeup, cleanse'], ['Wash pillowcases'], ['Squeezing pimples','Heavy oily cosmetics']),
 'urinary tract infection': _p('Hydration and hygiene.', ['2-3 L water daily','Wipe front to back','Urinate after intercourse','Do not hold urine'], ['Morning: water','Day: urinate regularly','Night: urinate before bed'], ['Wear cotton underwear, wash daily'], ['Holding urine','Harsh feminine washes']),
 'psoriasis': _p('Skin care and trigger control.', ['Moisturise daily','Manage stress','Moderate sunlight','Avoid skin injury and smoking'], ['Morning: lukewarm bath, moisturiser','Night: prescribed ointment'], ['Track flare triggers'], ['Scratching','Alcohol and smoking']),
 'impetigo': _p('Hygiene to stop spread.', ['Wash hands often','Keep sores covered','No sharing towels','Keep nails short'], ['Daily: gentle wash of sores, antibiotic cream as prescribed','Change clothes daily'], ['Hot wash towels and bedding'], ['Scratching','Going to school/work with open sores']),
}


def _key(name):
    return re.sub(r'\s+', ' ', str(name or '').strip().lower())


def get_preventive_plan(disease):
    """Returns prevention plan dict for a disease name (with defaults if unknown)."""
    k = _key(disease)
    for name, plan in _DATA.items():
        if _key(name) == k:
            return dict(plan, disease=str(disease).strip(), disclaimer='General wellness guidance, not medical advice. Consult a doctor.')
    return dict(_DEFAULT, disease=str(disease).strip(), disclaimer='General wellness guidance, not medical advice. Consult a doctor.')


def covered_diseases():
    return list(_DATA.keys())
