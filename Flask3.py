from flask import Flask, render_template, request, jsonify
from flask import send_from_directory
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import base64

app = Flask(__name__)

# Replace this with the actual class names used in your model
class_names = ["Acne and Rosacea Photos", "Actinic Keratosis  Carcinoma and Malignant Lesions",
               "Bullous Disease Photos", "Cellulitis Impetigo and Bacterial Infections",
               "Hair Loss Photos Alopecia and other Hair Diseases", "Nail Fungus and other Nail Disease",
               "Poison Ivy Photos and other Contact Dermatitis", "Vasculitis Photos"]
disease_info = {
    "Acne and Rosacea Photos": {
        "description": "Acne is a common skin condition that affects many people. It can be managed with proper skin care and medication.Rosacea is a chronic skin condition characterized by redness, visible blood vessels, and sometimes pimple-like bumps on the face. It can also cause a burning or stinging sensation. Rosacea primarily affects the central part of the face, such as the cheeks, nose, and forehead.",
        "tips": " Maintain a regular skincare routine and consult a dermatologist. Cleansing: Use a gentle, non-comedogenic (won't clog pores) cleanser to wash your face twice daily to remove excess oil and impurities.Topical Treatments: Over-the-counter or prescription acne treatments with ingredients like benzoyl peroxide, salicylic acid, or retinoids can help reduce breakouts",

    },
    "Actinic Keratosis  Carcinoma and Malignant Lesions": {
        "description": "Actinic keratosis, also known as solar keratosis or senile keratosis, is a pre-cancerous skin condition. It is characterized by rough, scaly patches of skin that often occur on sun-exposed areas, such as the face, ears, neck, scalp, chest, backs of hands, forearms, or lips. AK is a result of prolonged sun exposure and can develop over time.",
        "tips": "Protection: The best way to prevent AK is to protect your skin from excessive sun exposure. This includes wearing sunscreen, protective clothing (hats, long sleeves), and seeking shade during peak sun hours."
                "Regular Skin Checks: If you have a history of sun exposure, fair skin, or a family history of skin cancer, it's essential to conduct regular self-examinations and consult a dermatologist for check-ups.",
    },
    "Bullous Disease Photos": {
        "description": "Bullous pemphigoid is another autoimmune blistering disorder that primarily affects older individuals. It results in the formation of large, tense blisters, often on the arms, legs, and abdomen.",
        "tips": " Early diagnosis and treatment are important to prevent complications."
                "Corticosteroids, either topical or oral, are often prescribed to control inflammation and blister formation.",
    },
    "Cellulitis Impetigo and Bacterial Infections": {
        "description": "Cellulitis is a common bacterial skin infection that affects the deeper layers of the skin and the tissues beneath it. It typically occurs when bacteria enter the body through a break in the skin, such as a cut, scrape, or insect bite. Common causative bacteria include Streptococcus and Staphylococcus.Early recognition is crucial. Symptoms include redness, warmth, swelling, and tenderness at the affected site.",
        "tips": "If you suspect cellulitis, seek medical attention promptly. Left untreated, it can spread and become a more severe infection.",
    },
    "Hair Loss Photos Alopecia and other Hair Diseases": {
        "description": "Hair loss can also be associated with other hair diseases or disorders, such as telogen effluvium (temporary hair shedding), trichotillomania (compulsive hair-pulling disorder), and tinea capitis (a fungal infection of the scalp), among others.",
        "tips": "If you are experiencing significant hair loss, the first step is to consult a healthcare professional, such as a dermatologist or a trichologist. They can diagnose the specific cause of your hair loss and recommend appropriate treatments.",
    },
    "Nail Fungus and other Nail Disease": {
        "description": " Nail fungus, or onychomycosis, is a fungal infection that affects the nails, most commonly the toenails. It can cause nails to become discolored, thickened, brittle, and distorted. In some cases, the nail may detach from the nail bed.",
        "tips": "Keep your feet clean and dry. Wash them regularly, especially after sweating, and dry them thoroughly, especially between the toes.",
    },
    "Poison Ivy Photos and other Contact Dermatitis": {
        "description": "Poison Ivy (Toxicodendron radicans) is a common plant in North America that can cause a skin condition known as contact dermatitis when touched. Poison Ivy contains a resinous substance called urushiol, which is highly irritating to the skin.",
        "tips": "When in areas where poison ivy might be present, wear long sleeves, long pants, gloves, and closed-toe shoes. This reduces the risk of direct skin contact.",
    },

    "Vasculitis Photos": {
        "description": "There are many different types of vasculitis, classified based on the size of the blood vessels affected. Some common types include giant cell arteritis, Takayasu's arteritis, polyarteritis nodosa, and ANCA-associated vasculitis.",
        "tips": "If diagnosed with vasculitis, it's crucial to follow your healthcare provider's treatment plan diligently. This may include taking prescribed medications regularly and attending follow-up appointments to monitor your condition",
    },

}

# Replace this with your actual model
model = tf.keras.models.load_model('New3.h5')


def predict(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))  # Adjust the size as per your model's input shape
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence


@app.route('/')
def home3():
    return render_template('home3.html')
@app.route('/index3')
def index3():
    return render_template('index3.html')



@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        img_path = 'uploaded_image.jpg'  # Save the uploaded image temporarily
        file.save(img_path)
        predicted_class, confidence = predict(img_path)
        os.remove(img_path)  # Remove the temporary image file

        if predicted_class in disease_info:
            disease_details = disease_info[predicted_class]
            return jsonify({'predicted_class': predicted_class, 'confidence': confidence,
                            'description': disease_details['description'], 'tips': disease_details['tips']})
        else:
            return jsonify({'predicted_class': predicted_class, 'confidence': confidence,
                            'description': 'No information available.', 'tips': 'No information available.'})


if __name__ == '__main__':
    app.run(debug=True)
