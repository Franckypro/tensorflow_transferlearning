# Projet de Classification d'Images avec TensorFlow 🧠📸

## Description
Ce projet utilise **TensorFlow** pour effectuer de la **classification d'images** en utilisant un modèle préexistant de **ResNet50**. Le modèle est fine-tuné (apprentissage par transfert) pour être utilisé avec un ensemble de données local, permettant la classification binaire d'images, comme des photos de **chats** et **chiens**. 

## Technologies utilisées 🛠️
- **TensorFlow** (Version 2.16.1)
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **NumPy**

## Étapes du projet 🚀

### 1. Chargement et prétraitement de l'image 🖼️
L'image est chargée et redimensionnée à une taille de **224x224** pixels pour être compatible avec le modèle **ResNet50**.

```python
sample_image = tf.keras.preprocessing.image.load_img('bicycle.jpg', target_size=(224,224))
sample_image = tf.keras.preprocessing.image.img_to_array(sample_image)  # Transformation en tableau
sample_image = np.expand_dims(sample_image, axis=0)  # Préparation pour le modèle
sample_image = tf.keras.applications.resnet50.preprocess_input(sample_image)
```

### 2. Chargement du modèle pré-entraîné 🧑‍💻
Le modèle **ResNet50** est chargé avec des poids pré-entraînés sur **ImageNet**.

```python
model = tf.keras.applications.ResNet50(weights='imagenet')
```

### 3. Fine-tuning (Apprentissage par Transfert) 🔄
Le modèle **ResNet50** est chargé sans la partie supérieure (`include_top=False`), puis une série de couches denses est ajoutée pour adapter le modèle à la classification binaire (chats vs chiens).

```python
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
x = model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
# Ajouter plus de couches pour le fine-tuning
```

### 4. Entraînement du modèle ⚙️
Le modèle est entraîné sur un ensemble de données d'images locales avec un générateur d'images.

```python
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
train_generator = train_datagen.flow_from_directory('training_set', target_size=(224,224), batch_size=32, class_mode='binary')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(train_generator, epochs=5)
```

### 5. Visualisation des résultats 📊
Les courbes de **perte** et **précision** sont tracées pour évaluer l'évolution de l'entraînement.

```python
plt.plot(hist.history['accuracy'])
plt.title('Accuracy during training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy'])
plt.plot(hist.history['loss'])
plt.title('Loss during training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training Loss'])
```

### 6. Prédiction sur une nouvelle image 🐱🐶
Une fois l'entraînement terminé, le modèle peut être utilisé pour effectuer des prédictions sur de nouvelles images.

```python
sample_image = tf.keras.preprocessing.image.load_img('cat_pour_transfer.jpg', target_size=(224,224))
sample_image = tf.keras.preprocessing.image.img_to_array(sample_image)
sample_image = np.expand_dims(sample_image, axis=0)
sample_image = tf.keras.applications.resnet50.preprocess_input(sample_image)
predictions = model.predict(sample_image)
print('Prédictions :', predictions)
```

## Résultats 🎯
Le modèle sera capable de classifier les images en fonction des **chats** et **chiens** avec une précision croissante au fur et à mesure des époques d'entraînement.

## Prérequis 📦
- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- Pandas

## Conclusion 💡
Ce projet montre comment utiliser l'apprentissage par transfert pour adapter un modèle pré-entraîné à un problème spécifique, tout en mettant en œuvre des techniques courantes dans la classification d'images avec **TensorFlow**.

---

## ✍️ Auteur
Fouejio Francky Joël
