import { useEffect, useRef, useState } from "react"
import * as tf from "@tensorflow/tfjs";

function App() {
  const fileInputRef = useRef(null)
  const [model, setModel] = useState(null)
  const [loading, setLoading] = useState(true)
  const [imageUrl, setImageURL] = useState("https://www.placehold.co/250")
  const [predict, setPredict] = useState(false)
  const [predictText, setPredictText] = useState(null)
  const [fileName, setFileName] = useState("Choose an Image")

  useEffect(() => {
    async function loadModel() {
      try {
        const loadedModel = await tf.loadGraphModel("/Freshness-Detection-App/model/v4-resnet50/model.json");
        setModel(loadedModel);
      } catch (error) {
        console.error("Error loading model:", error);
      } finally {
        setLoading(false);
      }
    }

    loadModel();
  }, [])

  function handleChoose() {
    setPredict(false)
    fileInputRef.current.click()
  }

  function handleChange(e) {
    const file = e.target.files[0]

    if (file) {
      const imageURL = URL.createObjectURL(file);
      setImageURL(imageURL)
      setPredict(true)
      setFileName(file.name)
    }
  }

  async function predictImage() {
    if (!model || !imageUrl) return;

    try {
      setPredictText("Predicting...")
      const image = new Image();
      image.src = imageUrl;
      await new Promise((resolve) => {
        image.onload = resolve;
      });

      const inputTensor = tf.browser
        .fromPixels(image)
        .resizeBilinear([224, 224])
        .toFloat()
        .expandDims(0);


      
      // For Sigmoid & Binary Cross Entropy

      const output = await model.predict(inputTensor);

      const prediction = await output.data()

      const probability = prediction[0]

      const confidence = probability >= 0.5 ? 
      100 * probability : 
      100 * (1 - probability)

      let label = ""

      if (probability >= 0.5) {
        label = "Rotten"
      } else {
        label = "Fresh"
      }

      setPredictText(`${label} : ${confidence.toFixed(2)}%`)

      inputTensor.dispose();
      output.dispose();

    } catch (error) {
      console.error("Prediction error:", error);
      setPredictText("Prediction failed. An Error Occured.");
    }
  }


  return (
    <div className="container d-flex flex-column gap-3 align-items-center p-3">
      <h3 align="center">Freshness Detection for Fruits and Vegetables</h3>

      <p>{loading ? "Loading Model..." : "Model Loaded Successfully"}</p>
      <input type="file" accept="image/*" ref={fileInputRef} style={{ display: "none" }} onChange={handleChange} />
      <div className="container d-flex gap-2 align-items-center justify-content-center">
        <div className="input-img-label">{fileName}</div>
        <button className="btn btn-danger" onClick={handleChoose} disabled={loading}>Choose</button>
      </div>

      <img src={imageUrl} alt="Image" className="image" />

      <button className="btn btn-primary" disabled={!predict} onClick={predictImage}>Predict</button>

      <h2 className="m-2">{predictText}</h2>

    </div>
  )
}

export default App