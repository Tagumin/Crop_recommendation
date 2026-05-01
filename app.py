from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__, static_folder=".")
CORS(app)

# ========================
# Load model on startup
# ========================
MODEL_PATH = "crop_model.pkl"

saved = joblib.load(MODEL_PATH)
loaded_model = saved["model"]
saved_accuracy = saved["accuracy"]
saved_cols = saved["feature_cols"]

print(f"✅ Model loaded | Accuracy: {saved_accuracy * 100:.2f}%")

# ========================
# Input range validation
# ========================
VALID_RANGES = {
    "N":           (0,    140),
    "P":           (5,    145),
    "K":           (5,    205),
    "temperature": (8,    44),
    "humidity":    (14,   100),
    "ph":          (3.5,  9.5),
    "rainfall":    (20,   300),
}

CROP_INFO = {
    "rice":        {"planting": "Wet season (May–July)",          "harvest": "3–6 months after planting",    "care": "Requires flooded fields, high water supply, and fertile soil",        "pests": ["rice weevil", "stem borer"],         "diseases": ["blast", "bacterial leaf blight"]},
    "maize":       {"planting": "Early rainy season",             "harvest": "3–4 months",                   "care": "Needs well-drained soil and full sunlight",                           "pests": ["armyworm", "corn borer"],            "diseases": ["rust", "leaf blight"]},
    "chickpea":    {"planting": "Cool dry season",                "harvest": "3–5 months",                   "care": "Prefers dry climate and well-drained soil",                           "pests": ["pod borer"],                         "diseases": ["wilt", "root rot"]},
    "kidneybeans": {"planting": "Spring or early rainy season",   "harvest": "2–3 months",                   "care": "Needs moderate watering and good drainage",                           "pests": ["aphids"],                            "diseases": ["anthracnose", "rust"]},
    "pigeonpeas":  {"planting": "Rainy season",                   "harvest": "4–6 months",                   "care": "Drought-tolerant, grows in poor soil",                                "pests": ["pod fly"],                           "diseases": ["wilt"]},
    "mothbeans":   {"planting": "Rainy season",                   "harvest": "2–3 months",                   "care": "Thrives in dry regions with minimal water",                           "pests": ["beetles"],                           "diseases": ["leaf spot"]},
    "mungbean":    {"planting": "Warm season",                    "harvest": "2–3 months",                   "care": "Needs warm climate and light soil",                                   "pests": ["whiteflies"],                        "diseases": ["powdery mildew"]},
    "blackgram":   {"planting": "Rainy season",                   "harvest": "3–4 months",                   "care": "Prefers warm climate and moderate rainfall",                          "pests": ["aphids"],                            "diseases": ["leaf curl virus"]},
    "lentil":      {"planting": "Cool season",                    "harvest": "3–4 months",                   "care": "Needs well-drained soil and low humidity",                            "pests": ["aphids"],                            "diseases": ["rust"]},
    "pomegranate": {"planting": "Spring",                         "harvest": "5–7 months after flowering",   "care": "Needs dry climate and minimal irrigation",                            "pests": ["fruit borer"],                       "diseases": ["bacterial blight"]},
    "banana":      {"planting": "Anytime in tropical climate",    "harvest": "9–12 months",                  "care": "Requires rich soil, high humidity, and frequent watering",            "pests": ["weevils"],                           "diseases": ["panama disease"]},
    "mango":       {"planting": "Early rainy season",             "harvest": "3–5 years",                    "care": "Needs full sunlight and well-drained soil",                           "pests": ["fruit flies"],                       "diseases": ["anthracnose"]},
    "grapes":      {"planting": "Spring",                         "harvest": "1–3 years",                    "care": "Requires pruning, sunlight, and dry climate",                         "pests": ["mites"],                             "diseases": ["powdery mildew"]},
    "watermelon":  {"planting": "Warm season",                    "harvest": "2–3 months",                   "care": "Needs sandy soil and plenty of sunlight",                             "pests": ["aphids"],                            "diseases": ["fusarium wilt"]},
    "muskmelon":   {"planting": "Warm season",                    "harvest": "2–3 months",                   "care": "Requires well-drained soil and warm temperatures",                    "pests": ["fruit flies"],                       "diseases": ["downy mildew"]},
    "apple":       {"planting": "Winter/spring",                  "harvest": "4–5 years",                    "care": "Needs cold climate and pruning",                                      "pests": ["codling moth"],                      "diseases": ["apple scab"]},
    "orange":      {"planting": "Spring",                         "harvest": "2–3 years",                    "care": "Needs warm climate and regular watering",                             "pests": ["citrus psyllid"],                    "diseases": ["greening disease"]},
    "papaya":      {"planting": "Anytime in tropical climate",    "harvest": "6–9 months",                   "care": "Needs well-drained soil and frequent watering",                       "pests": ["aphids"],                            "diseases": ["papaya ringspot virus"]},
    "coconut":     {"planting": "Rainy season",                   "harvest": "4–6 years",                    "care": "Needs sandy soil and coastal climate",                                "pests": ["rhinoceros beetle"],                 "diseases": ["bud rot"]},
    "cotton":      {"planting": "Warm season",                    "harvest": "5–6 months",                   "care": "Requires full sunlight and moderate water",                           "pests": ["bollworm"],                          "diseases": ["wilt"]},
    "jute":        {"planting": "Rainy season",                   "harvest": "4–5 months",                   "care": "Needs high humidity and fertile soil",                                "pests": ["stem weevil"],                       "diseases": ["rot"]},
    "coffee":      {"planting": "Rainy season",                   "harvest": "3–4 years",                    "care": "Needs shade, high altitude, and regular rainfall",                    "pests": ["coffee borer"],                      "diseases": ["leaf rust"]},
}

CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🫘", "mothbeans": "🫘", "mungbean": "🫘", "blackgram": "🫘",
    "lentil": "🫘", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🍈", "coconut": "🥥", "cotton": "🌿",
    "jute": "🌿", "coffee": "☕",
}

PEST_INFO = {
    "aphids":           {"description": "Small sap-sucking insects that weaken plants and can transmit plant viruses.",                                    "symptoms": "Curled leaves, sticky honeydew, yellowing, stunted growth",                              "damage_level": "medium", "prevention": "Monitor plants regularly, encourage natural enemies, avoid excessive nitrogen",                   "treatment": "Use insecticidal soap, neem-based spray, or targeted insecticide if infestation is severe", "pesticides": ["Imidacloprid", "Acetamiprid", "Thiamethoxam"],                   "active_season": "Warm and dry seasons"},
    "armyworm":         {"description": "Caterpillars that feed aggressively on leaves and can destroy crops rapidly.",                                    "symptoms": "Chewed leaves, skeletonized foliage, sudden defoliation",                                "damage_level": "high",   "prevention": "Early field scouting, remove weeds, use pheromone traps",                                   "treatment": "Apply biological control or insecticide when larvae are young",                              "pesticides": ["Lambda-cyhalothrin", "Chlorantraniliprole", "Spinosad"],          "active_season": "Rainy season"},
    "beetles":          {"description": "A broad group of insects; many species feed on leaves, stems, roots, or stored crops.",                           "symptoms": "Holes in leaves, damaged roots, seed or grain loss",                                    "damage_level": "medium", "prevention": "Crop rotation, field sanitation, proper storage",                                            "treatment": "Use traps or apply insecticide depending on beetle species",                                 "pesticides": ["Carbaryl", "Cypermethrin", "Deltamethrin"],                       "active_season": "Warm season"},
    "bollworm":         {"description": "Larvae that attack cotton and fruiting crops by feeding on buds, flowers, and bolls.",                           "symptoms": "Damaged buds, bored bolls, reduced fruit set",                                           "damage_level": "high",   "prevention": "Use resistant varieties, destroy crop residues, monitor with traps",                        "treatment": "Apply insecticide when larvae are small",                                                    "pesticides": ["Emamectin benzoate", "Spinetoram", "Indoxacarb"],                 "active_season": "Warm season"},
    "codling moth":     {"description": "A major pest of apples whose larvae burrow into fruits and damage the core.",                                    "symptoms": "Fruit entry holes, frass near fruit surface, premature fruit drop",                     "damage_level": "high",   "prevention": "Use pheromone traps, remove fallen fruit, bag fruits if needed",                            "treatment": "Apply insecticide during moth flight period",                                                "pesticides": ["Lambda-cyhalothrin", "Chlorantraniliprole", "Spinosad"],          "active_season": "Spring to early summer"},
    "coffee borer":     {"description": "A beetle that bores into coffee berries and reduces bean quality and yield.",                                    "symptoms": "Small holes in berries, premature berry drop, low bean quality",                        "damage_level": "high",   "prevention": "Harvest on time, remove leftover berries, maintain field hygiene",                          "treatment": "Use traps and targeted control during infestation",                                          "pesticides": ["Chlorpyrifos", "Cypermethrin", "Endosulfan"],                     "active_season": "Rainy season"},
    "corn borer":       {"description": "Larvae that tunnel into maize stems and ears, weakening the plant.",                                            "symptoms": "Small holes in stems, broken stems, frass, poor ear development",                       "damage_level": "high",   "prevention": "Early planting, resistant varieties, remove crop residues",                                 "treatment": "Apply insecticide at early larval stage",                                                    "pesticides": ["Chlorantraniliprole", "Lambda-cyhalothrin", "Emamectin benzoate"],"active_season": "Rainy season"},
    "fruit borer":      {"description": "Larvae that bore into fruits and make them rot or become unmarketable.",                                         "symptoms": "Holes in fruits, rotting tissue, larval entry points",                                   "damage_level": "high",   "prevention": "Bag fruits, field sanitation, remove infested fruits",                                      "treatment": "Use pheromone traps and insecticide when needed",                                            "pesticides": ["Spinosad", "Lambda-cyhalothrin", "Indoxacarb"],                   "active_season": "Warm and humid seasons"},
    "fruit flies":      {"description": "Insects that lay eggs inside fruits; larvae feed internally and cause decay.",                                   "symptoms": "Soft spots, premature fruit drop, maggots inside fruit",                                 "damage_level": "high",   "prevention": "Use bait traps, collect fallen fruits, cover fruits",                                       "treatment": "Use protein bait sprays or targeted control",                                                "pesticides": ["Malathion", "Spinosad", "Lambda-cyhalothrin"],                    "active_season": "Warm and humid seasons"},
    "mites":            {"description": "Tiny pests that suck plant juices and cause leaf discoloration and stress.",                                     "symptoms": "Yellow speckling, bronzing, webbing, leaf drop",                                         "damage_level": "medium", "prevention": "Reduce dust, avoid water stress, monitor leaf undersides",                                  "treatment": "Apply miticide if infestation is severe",                                                    "pesticides": ["Abamectin", "Spiromesifen", "Propargite"],                        "active_season": "Hot and dry seasons"},
    "pod borer":        {"description": "Larvae that feed on pods of legumes and damage developing seeds.",                                               "symptoms": "Holes in pods, damaged seeds, premature pod drop",                                       "damage_level": "high",   "prevention": "Early monitoring, crop rotation, destroy crop residues",                                    "treatment": "Apply insecticide during flowering and pod formation",                                       "pesticides": ["Emamectin benzoate", "Spinosad", "Indoxacarb"],                   "active_season": "Warm season"},
    "pod fly":          {"description": "A pest that lays eggs in pods; larvae feed inside and damage the seeds.",                                        "symptoms": "Swollen pods, discolored seeds, reduced seed filling",                                   "damage_level": "medium", "prevention": "Timely sowing, field sanitation, destroy infected pods",                                    "treatment": "Use targeted insecticide during pod development",                                            "pesticides": ["Dimethoate", "Lambda-cyhalothrin", "Cypermethrin"],               "active_season": "Rainy season"},
    "rice weevil":      {"description": "Small beetle that damages stored rice grains and causes post-harvest losses.",                                   "symptoms": "Hollow grains, powdery residue, grain weight loss",                                      "damage_level": "high",   "prevention": "Store rice in sealed containers, keep humidity low",                                        "treatment": "Use grain protectants or fumigation for severe storage infestation",                         "pesticides": ["Malathion", "Pirimiphos-methyl", "Deltamethrin"],                 "active_season": "Year-round, worse in dry season"},
    "rhinoceros beetle":{"description": "A large beetle that attacks coconut palms by boring into the crown.",                                            "symptoms": "Cuts on young fronds, V-shaped cuts, reduced growth",                                    "damage_level": "high",   "prevention": "Remove breeding sites, maintain field sanitation, use traps",                               "treatment": "Apply insecticide to breeding sites or crown if needed",                                     "pesticides": ["Chlorpyrifos", "Imidacloprid", "Carbaryl"],                       "active_season": "Rainy season"},
    "stem borer":       {"description": "Caterpillar that bores into stems and disrupts nutrient flow in plants.",                                        "symptoms": "Dead heart in vegetative stage, white ear in reproductive stage",                        "damage_level": "high",   "prevention": "Use resistant varieties, adjust planting date, remove stubble",                             "treatment": "Apply granular insecticide at plant base or early larval stage",                             "pesticides": ["Carbofuran", "Chlorpyrifos", "Fipronil"],                         "active_season": "Rainy season"},
    "stem weevil":      {"description": "Insects that attack plant stems and can cause wilting and poor growth.",                                         "symptoms": "Stem tunnels, wilting, weakened stems, poor vigor",                                      "damage_level": "medium", "prevention": "Crop rotation, destroy residues, healthy planting material",                                "treatment": "Apply insecticide during early infestation",                                                 "pesticides": ["Imidacloprid", "Lambda-cyhalothrin", "Chlorpyrifos"],             "active_season": "Rainy season"},
    "weevils":          {"description": "A group of beetles that attack stored grains or field crops by feeding on seeds and plant tissue.",               "symptoms": "Small holes in grains, grain powder, reduced seed quality",                              "damage_level": "high",   "prevention": "Dry grains properly, store in sealed containers, clean storage areas",                      "treatment": "Use grain protectant or fumigation for storage infestation",                                 "pesticides": ["Malathion", "Pirimiphos-methyl", "Deltamethrin"],                 "active_season": "Year-round"},
    "whiteflies":       {"description": "Small flying insects that suck sap and spread plant viruses.",                                                   "symptoms": "Leaf yellowing, sticky honeydew, sooty mold, leaf curling",                              "damage_level": "medium", "prevention": "Use yellow sticky traps, remove weeds, avoid excess nitrogen",                              "treatment": "Apply insecticidal soap, neem spray, or systemic insecticide if severe",                    "pesticides": ["Imidacloprid", "Thiamethoxam", "Buprofezin"],                     "active_season": "Warm and dry seasons"},
    "citrus psyllid":   {"description": "A tiny jumping insect that feeds on citrus and is the main vector of citrus greening disease.",                  "symptoms": "Distorted new growth, waxy tubular secretions, yellowing shoots",                        "damage_level": "high",   "prevention": "Use certified disease-free nursery stock, apply reflective mulch, monitor regularly",       "treatment": "Apply systemic insecticide at flush stage; remove and destroy infected trees",               "pesticides": ["Imidacloprid", "Thiamethoxam", "Spirotetramat"],                  "active_season": "Warm and humid seasons"},
}

DISEASE_INFO = {
    "blast": {
        "description": "A destructive fungal disease of rice caused by Magnaporthe oryzae, capable of destroying entire fields under favorable conditions.",
        "symptoms": "Diamond-shaped lesions on leaves with gray center and brown border; neck rot causing white empty panicles in severe cases",
        "treatment": "Apply fungicide (Tricyclazole or Mancozeb) at first sign of symptoms; repeat application if conditions remain wet",
        "prevention": "Use resistant varieties, avoid excess nitrogen fertilizer, maintain proper plant spacing for airflow",
        "chemicals": ["Tricyclazole", "Mancozeb", "Azoxystrobin"],
        "damage_level": "high",
        "active_season": "Rainy season, high humidity"
    },
    "bacterial leaf blight": {
        "description": "A serious bacterial disease of rice caused by Xanthomonas oryzae pv. oryzae, especially damaging in flood-irrigated fields.",
        "symptoms": "Water-soaked to yellowish stripes on leaf margins that turn brown and dry; milky or yellow bacterial ooze on cut ends",
        "treatment": "Apply copper-based bactericide; drain fields, remove and destroy severely infected plant material",
        "prevention": "Use certified disease-free seeds, balanced nitrogen, ensure good field drainage",
        "chemicals": ["Copper hydroxide", "Copper oxychloride", "Streptomycin"],
        "damage_level": "high",
        "active_season": "Warm, rainy season"
    },
    "rust": {
        "description": "A fungal disease caused by Puccinia species that attacks leaves and stems, common across cereals and legumes.",
        "symptoms": "Orange-brown to reddish pustules on leaves and stems; premature leaf drop, reduced grain fill, and yield loss",
        "treatment": "Apply fungicide (Mancozeb or Propiconazole) at early infection; repeat every 14 days if conditions persist",
        "prevention": "Use resistant varieties, ensure good air circulation, avoid overhead irrigation",
        "chemicals": ["Mancozeb", "Propiconazole", "Tebuconazole"],
        "damage_level": "medium",
        "active_season": "Cool to warm, humid conditions"
    },
    "leaf blight": {
        "description": "A fungal or bacterial disease causing large-scale browning of leaf tissue, affecting many crop types.",
        "symptoms": "Large irregular brown or tan spots on leaves; blighting spreads from leaf tips and margins inward",
        "treatment": "Apply fungicide (Azoxystrobin or Mancozeb); remove and destroy severely infected leaves",
        "prevention": "Crop rotation, field sanitation after harvest, avoid plant wounding",
        "chemicals": ["Azoxystrobin", "Mancozeb", "Chlorothalonil"],
        "damage_level": "medium",
        "active_season": "Warm, wet weather"
    },
    "wilt": {
        "description": "A soilborne disease caused by Fusarium or Verticillium fungi that block plant vascular tissue and cause sudden collapse.",
        "symptoms": "Sudden wilting and yellowing of leaves, browning of internal stem tissue when cut, plant collapse and death",
        "treatment": "Remove and destroy infected plants; apply soil fungicide; solarize soil between seasons",
        "prevention": "Use resistant varieties, practice crop rotation, improve soil drainage, avoid waterlogging",
        "chemicals": ["Carbendazim", "Thiophanate-methyl", "Metalaxyl"],
        "damage_level": "high",
        "active_season": "Warm season, wet soils"
    },
    "root rot": {
        "description": "A soilborne disease caused by Phytophthora, Pythium, or Rhizoctonia that rots roots and kills plants from below.",
        "symptoms": "Darkened, mushy roots; yellowing leaves, stunted growth, plant wilting despite adequate moisture, plant collapse",
        "treatment": "Apply fungicide drench (Metalaxyl or Fosetyl-Al) to soil; remove infected plants promptly",
        "prevention": "Ensure well-drained soil, avoid overwatering, use disease-free planting material, treat seeds before planting",
        "chemicals": ["Metalaxyl", "Fosetyl-Al", "Mefenoxam"],
        "damage_level": "high",
        "active_season": "Wet and cool conditions"
    },
    "anthracnose": {
        "description": "A fungal disease caused by Colletotrichum species, affecting fruits, leaves, and stems in warm humid conditions.",
        "symptoms": "Dark sunken lesions on fruits, leaves, or stems; pink or salmon-colored spore masses appear in humid conditions",
        "treatment": "Apply fungicide (Mancozeb or Azoxystrobin); remove and destroy infected plant parts immediately",
        "prevention": "Avoid overhead irrigation, remove crop debris after harvest, use certified disease-free seeds",
        "chemicals": ["Mancozeb", "Azoxystrobin", "Difenoconazole"],
        "damage_level": "medium",
        "active_season": "Warm, humid seasons"
    },
    "leaf spot": {
        "description": "A fungal disease causing localized spotting on leaves, caused by various Cercospora or Alternaria species.",
        "symptoms": "Small circular to irregular brown or black spots on leaves, sometimes surrounded by a yellow halo; defoliation in severe cases",
        "treatment": "Apply fungicide (Chlorothalonil or Mancozeb) at early stage; remove and dispose of infected leaves",
        "prevention": "Avoid overcrowding plants, ensure good air circulation, remove plant debris after harvest",
        "chemicals": ["Chlorothalonil", "Mancozeb", "Iprodione"],
        "damage_level": "low",
        "active_season": "Warm and humid conditions"
    },
    "powdery mildew": {
        "description": "A fungal disease caused by Erysiphe or Podosphaera species that thrives in dry conditions with high humidity.",
        "symptoms": "White powdery fungal coating on leaf surfaces and shoots; curling, yellowing, and premature leaf drop",
        "treatment": "Apply sulfur-based fungicide or systemic fungicide (Triadimefon); neem oil effective for mild early cases",
        "prevention": "Ensure good air circulation, avoid excess humidity and dense planting, use resistant varieties",
        "chemicals": ["Sulfur", "Triadimefon", "Myclobutanil"],
        "damage_level": "medium",
        "active_season": "Dry weather with high humidity"
    },
    "leaf curl virus": {
        "description": "A viral disease transmitted by whiteflies (Bemisia tabaci), causing severe deformation of leaf tissue.",
        "symptoms": "Upward curling and distortion of leaves, intense yellowing, severely stunted plant growth and reduced yield",
        "treatment": "No chemical cure for the virus; remove and destroy infected plants immediately; aggressively control whitefly vectors",
        "prevention": "Use virus-free certified seeds, control whitefly populations with insecticides and yellow sticky traps, use reflective mulch",
        "chemicals": ["Imidacloprid (vector control)", "Thiamethoxam (vector control)"],
        "damage_level": "high",
        "active_season": "Warm, dry seasons — peak whitefly activity"
    },
    "bacterial blight": {
        "description": "A bacterial disease caused by Xanthomonas or Pseudomonas species, affecting a wide range of fruit and vegetable crops.",
        "symptoms": "Water-soaked lesions that turn dark brown and necrotic; gummy bacterial exudate on infected fruits or stems",
        "treatment": "Apply copper-based bactericide; prune infected branches and sterilize pruning tools between cuts",
        "prevention": "Use certified disease-free planting material, avoid plant injuries, maintain strict field sanitation",
        "chemicals": ["Copper hydroxide", "Copper oxychloride", "Streptomycin sulfate"],
        "damage_level": "medium",
        "active_season": "Warm, rainy season"
    },
    "panama disease": {
        "description": "A devastating soilborne Fusarium wilt disease of banana caused by Fusarium oxysporum f.sp. cubense. No effective cure exists.",
        "symptoms": "Yellowing of older outer leaves progressing inward, wilting, dark brown vascular discoloration inside the pseudostem",
        "treatment": "No effective chemical treatment; remove and destroy all infected plants; solarize soil and allow long fallow period",
        "prevention": "Plant Fusarium-resistant Cavendish varieties, prevent movement of infected soil, maintain proper drainage",
        "chemicals": ["No effective chemical — prevention is critical"],
        "damage_level": "high",
        "active_season": "Year-round in tropical climates"
    },
    "fusarium wilt": {
        "description": "A soilborne fungal disease caused by Fusarium oxysporum that infects the vascular system and is highly persistent in soil.",
        "symptoms": "One-sided yellowing and wilting of leaves, brown discoloration of vascular tissue when stem is cut, plant death",
        "treatment": "Apply soil fungicide (Carbendazim); remove infected plants and solarize soil to reduce inoculum",
        "prevention": "Use resistant varieties, practice long crop rotations, improve soil drainage, adjust soil pH to 6.5–7.0",
        "chemicals": ["Carbendazim", "Thiophanate-methyl", "Trichoderma-based biocontrol"],
        "damage_level": "high",
        "active_season": "Warm season, waterlogged soils"
    },
    "downy mildew": {
        "description": "A fungal-like disease caused by Plasmopara or Pseudoperonospora species that thrives in cool, wet conditions.",
        "symptoms": "Yellow patches on upper leaf surface with grayish-purple downy mold growth on the underside; rapid spread in wet weather",
        "treatment": "Apply fungicide (Metalaxyl or Fosetyl-Al) at first symptoms; improve air circulation and reduce leaf wetness",
        "prevention": "Avoid overhead irrigation, plant in well-drained soil, use resistant varieties, space plants adequately",
        "chemicals": ["Metalaxyl", "Fosetyl-Al", "Dimethomorph"],
        "damage_level": "medium",
        "active_season": "Cool, wet conditions"
    },
    "apple scab": {
        "description": "A fungal disease of apple caused by Venturia inaequalis, causing cosmetic and structural damage to fruit and foliage.",
        "symptoms": "Olive-green to dark brown scabby lesions on leaves and fruit surfaces; premature leaf and fruit drop in severe cases",
        "treatment": "Apply fungicide (Captan or Myclobutanil) from bud break through early fruit development; repeat every 7–14 days",
        "prevention": "Remove and destroy fallen leaves in autumn, use resistant apple varieties, ensure good canopy ventilation through pruning",
        "chemicals": ["Captan", "Myclobutanil", "Thiram"],
        "damage_level": "medium",
        "active_season": "Spring, wet weather at bud break"
    },
    "greening disease": {
        "description": "Also known as Huanglongbing (HLB), a deadly bacterial disease of citrus spread by the Asian citrus psyllid. No cure exists.",
        "symptoms": "Asymmetric blotchy yellowing of leaves, small bitter misshapen fruits, dieback of branches, eventual tree death",
        "treatment": "No cure; remove and destroy infected trees immediately; aggressively control Asian citrus psyllid vector population",
        "prevention": "Use certified disease-free nursery stock, implement strict psyllid monitoring and control, quarantine infected areas",
        "chemicals": ["Imidacloprid (psyllid control)", "Thiamethoxam (psyllid control)"],
        "damage_level": "high",
        "active_season": "Year-round in warm tropical and subtropical regions"
    },
    "papaya ringspot virus": {
        "description": "A devastating viral disease of papaya transmitted by aphids, causing fruit and foliage deformation and severe yield loss.",
        "symptoms": "Yellow mosaic patterns on leaves, ring-shaped spots on fruits, oily water-soaked streaks on stems, severely distorted growth",
        "treatment": "No chemical cure; remove and destroy infected plants immediately; control aphid vector populations aggressively",
        "prevention": "Use virus-free certified seedlings, apply insecticides to control aphid vectors, plant barrier crops of non-host plants",
        "chemicals": ["Imidacloprid (aphid control)", "Mineral oil spray (aphid deterrent)"],
        "damage_level": "high",
        "active_season": "Year-round in tropical climates"
    },
    "bud rot": {
        "description": "A serious disease of coconut palm caused by Phytophthora palmivora that destroys the growing point of the tree.",
        "symptoms": "Rotting and collapse of the central growing bud (spear leaf), foul odor, water-soaked base of spear, tree death if untreated",
        "treatment": "Apply copper-based fungicide directly to the crown; surgically remove and destroy all infected tissue",
        "prevention": "Avoid mechanical injury to the crown, improve drainage around palms, reduce excess moisture and standing water",
        "chemicals": ["Copper oxychloride", "Metalaxyl", "Fosetyl-Al"],
        "damage_level": "high",
        "active_season": "Wet, humid conditions"
    },
    "rot": {
        "description": "A broad category of fungal or bacterial decay affecting stems, roots, or stored plant material under wet conditions.",
        "symptoms": "Soft, discolored, and foul-smelling decay on stems, roots, or harvested plant material; structural collapse of tissue",
        "treatment": "Apply appropriate fungicide or bactericide; remove and destroy all infected plant material promptly",
        "prevention": "Ensure proper drainage and avoid waterlogging, practice crop rotation and field sanitation, harvest at correct maturity",
        "chemicals": ["Mancozeb", "Copper oxychloride", "Propamocarb"],
        "damage_level": "medium",
        "active_season": "Wet season, waterlogged soils"
    },
    "leaf rust": {
        "description": "A fungal disease of coffee caused by Hemileia vastatrix, considered the most economically damaging coffee disease worldwide.",
        "symptoms": "Pale yellow to orange powdery pustules on the underside of leaves; premature defoliation and severe yield reduction",
        "treatment": "Apply fungicide (Triadimefon or Propiconazole) at the onset of infection; repeat applications during the wet season",
        "prevention": "Use rust-resistant coffee varieties, manage shade and plant density, ensure adequate nutrition to strengthen plant immunity",
        "chemicals": ["Triadimefon", "Propiconazole", "Copper hydroxide"],
        "damage_level": "medium",
        "active_season": "Rainy season, high humidity"
    },
}


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Field '{field}' not found"}), 400

        sample_dict = {k: float(data[k]) for k in required}

        errors = []
        for feature, value in sample_dict.items():
            low, high = VALID_RANGES[feature]
            if not (low <= value <= high):
                errors.append(f"{feature} = {value} (must be between {low}–{high})")

        if errors:
            return jsonify({"error": "Invalid input", "details": errors}), 422

        sample_df = pd.DataFrame([sample_dict])[saved_cols]
        result = loaded_model.predict(sample_df)[0]
        proba = round(loaded_model.predict_proba(sample_df).max() * 100, 2)
        emoji = CROP_EMOJI.get(result.lower(), "🌱")

        return jsonify({
            "crop": result,
            "emoji": emoji,
            "confidence": proba,
            "model_accuracy": round(saved_accuracy * 100, 2)
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/crop-info/<crop_name>")
def crop_info(crop_name):
    info = CROP_INFO.get(crop_name.lower())
    if not info:
        return jsonify({"error": f"Info for '{crop_name}' not found"}), 404
    return jsonify(info)


@app.route("/pest-info/<path:pest_name>")
def pest_info(pest_name):
    info = PEST_INFO.get(pest_name.lower())
    if not info:
        return jsonify({"error": f"Info for '{pest_name}' not found"}), 404
    return jsonify(info)


@app.route("/disease-info/<path:disease_name>")
def disease_info(disease_name):
    info = DISEASE_INFO.get(disease_name.lower())
    if not info:
        return jsonify({"error": f"Info for '{disease_name}' not found"}), 404
    return jsonify(info)


@app.route("/pest_image/<path:filename>")
def pest_image(filename):
    return send_from_directory("pest_image", filename)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_accuracy": round(saved_accuracy * 100, 2),
        "features": saved_cols
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)