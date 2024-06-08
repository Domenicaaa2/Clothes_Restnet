<script>
  import { onMount } from 'svelte';
  let imageFile;
  let imageUrl;
  let predictedClass = null;
  let errorMessage = null;
  let selectedModel = 'First run CNN';
  let probabilities = [];
  let modelDetails = {
    'First run CNN': {
      description: 'Initial model with basic CNN architecture.',
      details: 'Epochs: 10, Learning Rate: 0.001, Optimizer: Adam, Loss: categorical_crossentropy'
    },
    'Second run adjustments made': {
      description: 'Used VGG16 model with transfer learning and fine-tuning, adjusted class weights.',
      details: 'Epochs: 10, Learning Rate: 0.0001, Optimizer: Adam, Loss: categorical_crossentropy',
      extra: 'VGG16 is a pre-trained convolutional neural network model used for image classification tasks, known for its depth and performance.'
    },
    'Third run': {
      description: 'Deep CNN model with additional dropout and batch normalization layers for improved generalization.',
      details: 'Epochs: 10, Learning Rate: 0.0001, Optimizer: Adam, Loss: categorical_crossentropy'
    },
    'Fourth run': {
      description: 'Transfer learning with VGG16 and additional layers including dropout for improved performance on custom dataset.',
      details: 'Epochs: 15, Learning Rate: 0.0001, Optimizer: Adam, Loss: categorical_crossentropy',
      extra: 'This model includes further fine-tuning on a custom dataset with increased data augmentation and early stopping.'
    }
  };

  const class_names = ['Apparel', 'Accessories', 'Footwear', 'Personal Care', 'Free Items', 'Sporting Goods', 'Home'];

  const handleFileChange = (event) => {
    const files = event.target.files;
    if (files.length > 0) {
      imageFile = files[0];
      imageUrl = URL.createObjectURL(imageFile);
    }
  };

  const handleModelChange = (event) => {
    selectedModel = event.target.value;
  };

  const handleSubmit = async () => {
    if (!imageFile) {
      errorMessage = "Please select an image file.";
      return;
    }

    const formData = new FormData();
    formData.append("file", imageFile);
    formData.append("model", selectedModel);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      if (result.error) {
        errorMessage = result.error;
      } else {
        predictedClass = result.predicted_class;
        probabilities = result.probabilities[0];
      }
    } catch (error) {
      errorMessage = "An error occurred while processing the request.";
    }
  };
</script>

<div class="container">
  <div class="left-side"></div>
  <div class="right-side">
    <h1>Fashion Classifier</h1>
    <h2>Upload your fashion item and get a prediction for a mastercategory and the article type</h2>
    <input type="file" accept="image/*" id="file-input" on:change="{handleFileChange}">
    <label for="file-input">Choose Image</label>
    
    <label for="model-select">Choose Model:</label>
    <select id="model-select" on:change="{handleModelChange}">
      <option value="First run CNN">First run CNN</option>
      <option value="Second run adjustments made">Second run adjustments made</option>
      <option value="Third run">Third run</option>
      <option value="Fourth run">Fourth run</option>
    </select>
    
    <p>{modelDetails[selectedModel].description}</p>
    <p>{modelDetails[selectedModel].details}</p>
    {#if modelDetails[selectedModel].extra}
      <p>{modelDetails[selectedModel].extra}</p>
    {/if}

    <button on:click="{handleSubmit}">Predict</button>

    {#if errorMessage}
      <p style="color: red;">{errorMessage}</p>
    {/if}

    {#if imageUrl}
      <img src="{imageUrl}" alt="Uploaded Image">
    {/if}

    {#if predictedClass !== null}
      <p class="prediction">Predicted Class: {predictedClass}</p>
      <h3>Prediction Probabilities:</h3>
      <ul>
        {#each probabilities as probability, index}
          <li>{class_names[index]}: {Math.round(probability * 100)}%</li>
        {/each}
      </ul>
    {/if}
  </div>
</div>
