from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__, static_folder=".")
CORS(app)

# ========================
# Load model saat startup
# ========================
MODEL_PATH = "crop_model.pkl"

saved = joblib.load(MODEL_PATH)
loaded_model = saved["model"]
saved_accuracy = saved["accuracy"]
saved_cols = saved["feature_cols"]

print(f"✅ Model loaded | Accuracy: {saved_accuracy * 100:.2f}%")

# ========================
# Validasi range input
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
    "rice": {"planting": "Wet season (May–July)", "harvest": "3–6 months after planting", "care": "Requires flooded fields, high water supply, and fertile soil", "pests": ["rice weevil", "stem borer"], "diseases": ["blast", "bacterial leaf blight"]},
    "maize": {"planting": "Early rainy season", "harvest": "3–4 months", "care": "Needs well-drained soil and full sunlight", "pests": ["armyworm", "corn borer"], "diseases": ["rust", "leaf blight"]},
    "chickpea": {"planting": "Cool dry season", "harvest": "3–5 months", "care": "Prefers dry climate and well-drained soil", "pests": ["pod borer"], "diseases": ["wilt", "root rot"]},
    "kidneybeans": {"planting": "Spring or early rainy season", "harvest": "2–3 months", "care": "Needs moderate watering and good drainage", "pests": ["aphids"], "diseases": ["anthracnose", "rust"]},
    "pigeonpeas": {"planting": "Rainy season", "harvest": "4–6 months", "care": "Drought-tolerant, grows in poor soil", "pests": ["pod fly"], "diseases": ["wilt"]},
    "mothbeans": {"planting": "Rainy season", "harvest": "2–3 months", "care": "Thrives in dry regions with minimal water", "pests": ["beetles"], "diseases": ["leaf spot"]},
    "mungbean": {"planting": "Warm season", "harvest": "2–3 months", "care": "Needs warm climate and light soil", "pests": ["whiteflies"], "diseases": ["powdery mildew"]},
    "blackgram": {"planting": "Rainy season", "harvest": "3–4 months", "care": "Prefers warm climate and moderate rainfall", "pests": ["aphids"], "diseases": ["leaf curl virus"]},
    "lentil": {"planting": "Cool season", "harvest": "3–4 months", "care": "Needs well-drained soil and low humidity", "pests": ["aphids"], "diseases": ["rust"]},
    "pomegranate": {"planting": "Spring", "harvest": "5–7 months after flowering", "care": "Needs dry climate and minimal irrigation", "pests": ["fruit borer"], "diseases": ["bacterial blight"]},
    "banana": {"planting": "Anytime in tropical climate", "harvest": "9–12 months", "care": "Requires rich soil, high humidity, and frequent watering", "pests": ["weevils"], "diseases": ["Panama disease"]},
    "mango": {"planting": "Early rainy season", "harvest": "3–5 years", "care": "Needs full sunlight and well-drained soil", "pests": ["fruit flies"], "diseases": ["anthracnose"]},
    "grapes": {"planting": "Spring", "harvest": "1–3 years", "care": "Requires pruning, sunlight, and dry climate", "pests": ["mites"], "diseases": ["powdery mildew"]},
    "watermelon": {"planting": "Warm season", "harvest": "2–3 months", "care": "Needs sandy soil and plenty of sunlight", "pests": ["aphids"], "diseases": ["fusarium wilt"]},
    "muskmelon": {"planting": "Warm season", "harvest": "2–3 months", "care": "Requires well-drained soil and warm temperatures", "pests": ["fruit flies"], "diseases": ["downy mildew"]},
    "apple": {"planting": "Winter/spring", "harvest": "4–5 years", "care": "Needs cold climate and pruning", "pests": ["codling moth"], "diseases": ["apple scab"]},
    "orange": {"planting": "Spring", "harvest": "2–3 years", "care": "Needs warm climate and regular watering", "pests": ["citrus psyllid"], "diseases": ["greening disease"]},
    "papaya": {"planting": "Anytime in tropical climate", "harvest": "6–9 months", "care": "Needs well-drained soil and frequent watering", "pests": ["aphids"], "diseases": ["papaya ringspot virus"]},
    "coconut": {"planting": "Rainy season", "harvest": "4–6 years", "care": "Needs sandy soil and coastal climate", "pests": ["rhinoceros beetle"], "diseases": ["bud rot"]},
    "cotton": {"planting": "Warm season", "harvest": "5–6 months", "care": "Requires full sunlight and moderate water", "pests": ["bollworm"], "diseases": ["wilt"]},
    "jute": {"planting": "Rainy season", "harvest": "4–5 months", "care": "Needs high humidity and fertile soil", "pests": ["stem weevil"], "diseases": ["rot"]},
    "coffee": {"planting": "Rainy season", "harvest": "3–4 years", "care": "Needs shade, high altitude, and regular rainfall", "pests": ["coffee borer"], "diseases": ["leaf rust"]},
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
    "aphids": {"description": "Small sap-sucking insects that weaken plants and can transmit plant viruses.", "symptoms": "Curled leaves, sticky honeydew, yellowing, stunted growth", "damage_level": "medium", "prevention": "Monitor plants regularly, encourage natural enemies, avoid excessive nitrogen", "treatment": "Use insecticidal soap, neem-based spray, or targeted insecticide if infestation is severe", "pesticides": ["Imidacloprid", "Acetamiprid", "Thiamethoxam"], "active_season": "Warm and dry seasons"},
    "armyworm": {"description": "Caterpillars that feed aggressively on leaves and can destroy crops rapidly.", "symptoms": "Chewed leaves, skeletonized foliage, sudden defoliation", "damage_level": "high", "prevention": "Early field scouting, remove weeds, use pheromone traps", "treatment": "Apply biological control or insecticide when larvae are young", "pesticides": ["Lambda-cyhalothrin", "Chlorantraniliprole", "Spinosad"], "active_season": "Rainy season"},
    "beetles": {"description": "A broad group of insects; many species feed on leaves, stems, roots, or stored crops.", "symptoms": "Holes in leaves, damaged roots, seed or grain loss", "damage_level": "medium", "prevention": "Crop rotation, field sanitation, proper storage", "treatment": "Use traps or apply insecticide depending on beetle species", "pesticides": ["Carbaryl", "Cypermethrin", "Deltamethrin"], "active_season": "Warm season"},
    "bollworm": {"description": "Larvae that attack cotton and fruiting crops by feeding on buds, flowers, and bolls.", "symptoms": "Damaged buds, bored bolls, reduced fruit set", "damage_level": "high", "prevention": "Use resistant varieties, destroy crop residues, monitor with traps", "treatment": "Apply insecticide when larvae are small", "pesticides": ["Emamectin benzoate", "Spinetoram", "Indoxacarb"], "active_season": "Warm season"},
    "codling moth": {"description": "A major pest of apples whose larvae burrow into fruits and damage the core.", "symptoms": "Fruit entry holes, frass near fruit surface, premature fruit drop", "damage_level": "high", "prevention": "Use pheromone traps, remove fallen fruit, bag fruits if needed", "treatment": "Apply insecticide during moth flight period", "pesticides": ["Lambda-cyhalothrin", "Chlorantraniliprole", "Spinosad"], "active_season": "Spring to early summer"},
    "coffee borer": {"description": "A beetle that bores into coffee berries and reduces bean quality and yield.", "symptoms": "Small holes in berries, premature berry drop, low bean quality", "damage_level": "high", "prevention": "Harvest on time, remove leftover berries, maintain field hygiene", "treatment": "Use traps and targeted control during infestation", "pesticides": ["Chlorpyrifos", "Cypermethrin", "Endosulfan"], "active_season": "Rainy season"},
    "corn borer": {"description": "Larvae that tunnel into maize stems and ears, weakening the plant.", "symptoms": "Small holes in stems, broken stems, frass, poor ear development", "damage_level": "high", "prevention": "Early planting, resistant varieties, remove crop residues", "treatment": "Apply insecticide at early larval stage", "pesticides": ["Chlorantraniliprole", "Lambda-cyhalothrin", "Emamectin benzoate"], "active_season": "Rainy season"},
    "fruit borer": {"description": "Larvae that bore into fruits and make them rot or become unmarketable.", "symptoms": "Holes in fruits, rotting tissue, larval entry points", "damage_level": "high", "prevention": "Bag fruits, field sanitation, remove infested fruits", "treatment": "Use pheromone traps and insecticide when needed", "pesticides": ["Spinosad", "Lambda-cyhalothrin", "Indoxacarb"], "active_season": "Warm and humid seasons"},
    "fruit flies": {"description": "Insects that lay eggs inside fruits; larvae feed internally and cause decay.", "symptoms": "Soft spots, premature fruit drop, maggots inside fruit", "damage_level": "high", "prevention": "Use bait traps, collect fallen fruits, cover fruits", "treatment": "Use protein bait sprays or targeted control", "pesticides": ["Malathion", "Spinosad", "Lambda-cyhalothrin"], "active_season": "Warm and humid seasons"},
    "mites": {"description": "Tiny pests that suck plant juices and cause leaf discoloration and stress.", "symptoms": "Yellow speckling, bronzing, webbing, leaf drop", "damage_level": "medium", "prevention": "Reduce dust, avoid water stress, monitor leaf undersides", "treatment": "Apply miticide if infestation is severe", "pesticides": ["Abamectin", "Spiromesifen", "Propargite"], "active_season": "Hot and dry seasons"},
    "pod borer": {"description": "Larvae that feed on pods of legumes and damage developing seeds.", "symptoms": "Holes in pods, damaged seeds, premature pod drop", "damage_level": "high", "prevention": "Early monitoring, crop rotation, destroy crop residues", "treatment": "Apply insecticide during flowering and pod formation", "pesticides": ["Emamectin benzoate", "Spinosad", "Indoxacarb"], "active_season": "Warm season"},
    "pod fly": {"description": "A pest that lays eggs in pods; larvae feed inside and damage the seeds.", "symptoms": "Swollen pods, discolored seeds, reduced seed filling", "damage_level": "medium", "prevention": "Timely sowing, field sanitation, destroy infected pods", "treatment": "Use targeted insecticide during pod development", "pesticides": ["Dimethoate", "Lambda-cyhalothrin", "Cypermethrin"], "active_season": "Rainy season"},
    "rice weevil": {"description": "Small beetle that damages stored rice grains and causes post-harvest losses.", "symptoms": "Hollow grains, powdery residue, grain weight loss", "damage_level": "high", "prevention": "Store rice in sealed containers, keep humidity low", "treatment": "Use grain protectants or fumigation for severe storage infestation", "pesticides": ["Malathion", "Pirimiphos-methyl", "Deltamethrin"], "active_season": "Year-round, worse in dry season"},
    "rhinoceros beetle": {"description": "A large beetle that attacks coconut palms by boring into the crown.", "symptoms": "Cuts on young fronds, V-shaped cuts, reduced growth", "damage_level": "high", "prevention": "Remove breeding sites, maintain field sanitation, use traps", "treatment": "Apply insecticide to breeding sites or crown if needed", "pesticides": ["Chlorpyrifos", "Imidacloprid", "Carbaryl"], "active_season": "Rainy season"},
    "stem borer": {"description": "Caterpillar that bores into stems and disrupts nutrient flow in plants.", "symptoms": "Dead heart in vegetative stage, white ear in reproductive stage", "damage_level": "high", "prevention": "Use resistant varieties, adjust planting date, remove stubble", "treatment": "Apply granular insecticide at plant base or early larval stage", "pesticides": ["Carbofuran", "Chlorpyrifos", "Fipronil"], "active_season": "Rainy season"},
    "stem weevil": {"description": "Insects that attack plant stems and can cause wilting and poor growth.", "symptoms": "Stem tunnels, wilting, weakened stems, poor vigor", "damage_level": "medium", "prevention": "Crop rotation, destroy residues, healthy planting material", "treatment": "Apply insecticide during early infestation", "pesticides": ["Imidacloprid", "Lambda-cyhalothrin", "Chlorpyrifos"], "active_season": "Rainy season"},
    "weevils": {"description": "A group of beetles that attack stored grains or field crops by feeding on seeds and plant tissue.", "symptoms": "Small holes in grains, grain powder, reduced seed quality", "damage_level": "high", "prevention": "Dry grains properly, store in sealed containers, clean storage areas", "treatment": "Use grain protectant or fumigation for storage infestation", "pesticides": ["Malathion", "Pirimiphos-methyl", "Deltamethrin"], "active_season": "Year-round"},
    "whiteflies": {"description": "Small flying insects that suck sap and spread plant viruses.", "symptoms": "Leaf yellowing, sticky honeydew, sooty mold, leaf curling", "damage_level": "medium", "prevention": "Use yellow sticky traps, remove weeds, avoid excess nitrogen", "treatment": "Apply insecticidal soap, neem spray, or systemic insecticide if severe", "pesticides": ["Imidacloprid", "Thiamethoxam", "Buprofezin"], "active_season": "Warm and dry seasons"},
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
                return jsonify({"error": f"Field '{field}' tidak ditemukan"}), 400

        # Cast ke float
        sample_dict = {k: float(data[k]) for k in required}

        # Validasi range
        errors = []
        for feature, value in sample_dict.items():
            low, high = VALID_RANGES[feature]
            if not (low <= value <= high):
                errors.append(f"{feature} = {value} (harus antara {low}–{high})")

        if errors:
            return jsonify({"error": "Input tidak valid", "details": errors}), 422

        # Prediksi
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
        return jsonify({"error": f"Nilai tidak valid: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/crop-info/<crop_name>")
def crop_info(crop_name):
    info = CROP_INFO.get(crop_name.lower())
    if not info:
        return jsonify({"error": f"Info untuk '{crop_name}' tidak ditemukan"}), 404
    return jsonify(info)


@app.route("/pest-info/<path:pest_name>")
def pest_info(pest_name):
    info = PEST_INFO.get(pest_name.lower())
    if not info:
        return jsonify({"error": f"Info untuk '{pest_name}' tidak ditemukan"}), 404
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