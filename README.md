# PromptScene: Adaptive Prompt Learning for Open-Vocabulary 3D Instance Segmentation

## Project Overview
This project was developed as part of the 'Machine Learning for 3D Geometry' course at the Technical University of Munich.

PromptScene is a machine learning pipeline for open-vocabulary 3D instance segmentation. It wasOur approach processes point cloud data and utilizes per-point features extracted from OpenScene. These features are aggregated per instance using instance masks predicted by Mask3D. Additionally, we implement prompt learning to improve segmentation accuracy by combining ground truth masks with OpenScene features.

## Pipeline Overview
1. **Input:** Point cloud data.
2. **Feature Extraction:** OpenScene extracts per-point features.
3. **Instance Segmentation:** Mask3D predicts instance masks.
4. **Feature Aggregation:** Per-instance feature averaging based on Mask3D predictions.
5. **Prompt Learning:** Combines ground truth masks with OpenScene features to enhance accuracy.
6. **Inference and Evaluation:** Perform segmentation on new data and analyze results.

## File Structure
```
├── config
│   └── ...                # Configuration files for OpenScene
├── dataset
│   └── data               # Contains the dataset used for training and evaluation
├── experiments
│   ├── run_...            # Generated results from different experiments
├── models
│   ├── openscene          # OpenScene feature extraction model
│   ├── mask3d             # Mask3D instance segmentation model
│   ├── multimodal-prompt-learning    # Implementation of prompt learning for segmentation
├── promptscene.ipynb  # Main notebook implementing the pipeline
├── inference.ipynb    # Notebook for inference on new data
├── evaluation.ipynb   # Notebook for evaluating model performance
├── README.md              # Project documentation
```

## Key Files
- **promptscene.ipynb**: Implements the full PromptScene pipeline.
- **inference.ipynb**: Runs inference on new point cloud data.
- **evaluation.ipynb**: Evaluates model performance using various metrics.
- **models/**: Contains the required models (OpenScene, Mask3D, and Prompt Learning).
- **dataset/data/**: Stores the dataset required for training and evaluation.
- **experiments/run_.../**: Contains results from different experimental runs.
- **config/**: Configuration files for OpenScene setup.

## Replicating Results
1. Clone the repository and navigate to the project directory:
   ```sh
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install required dependencies following the instructions in promptscene.ipynb and models/mask3d/mask3d.ipynb (ensure OpenScene and Mask3D are properly set up).
3. Run `promptscene.ipynb` to execute the pipeline.
4. Use `inference.ipynb` to segment new data.
5. Evaluate performance using `evaluation.ipynb`.
6. Check results in the `experiments/` folder.

## Conclusion
PromptScene demonstrates the effectiveness of adaptive prompt learning in open-vocabulary 3D instance segmentation. By leveraging OpenScene features, Mask3D instance masks, and prompt learning, we enhance segmentation accuracy on 3D point clouds.

