---
name: Feature request
about: Suggest a new feature, improvement, or enhancement for TC-MMWM
title: "[FEATURE] Real-Time Counterfactual Trajectory Visualization"
labels: enhancement
assignees: ''
---

# Feature Request

## Summary
I would like to request a feature to visualize counterfactual action rollouts in real-time during robot execution. Currently, counterfactual trajectories are only accessible via offline analysis in Figure 3 and Figure 5 notebooks. A real-time overlay of predicted future states on camera inputs would greatly enhance interpretability, debugging, and user feedback for long-horizon robotic tasks.

---

## Proposed Implementation
The feature could be implemented as follows:

1. **Visualization Utility**  
   - Extend `tc_mmwm/utils/visualization.py` with a new module `visualize_counterfactual_rollouts` that receives latent states and predicted actions.  
   - Overlay the predicted object trajectories onto RGB frames from the robot camera.  
   - Use different colors for multiple candidate actions for clear differentiation.

2. **Integration with Realtime Inference**  
   - Modify `scripts/realtime_inference.py` to optionally enable real-time rendering of counterfactual predictions.  
   - Include a configuration flag in `configs/deployment/*.yaml` files (`visualize_counterfactual: true/false`) for easy toggling.  

3. **Hardware Considerations**  
   - Implement GPU-accelerated rendering for RTX and Jetson platforms.  
   - Ensure minimal impact on inference latency (<5 ms added to per-step computation).  

4. **Example Notebooks**  
   - Provide an example notebook (`examples/realtime_counterfactual.ipynb`) demonstrating real-time visualization on simulated tasks, mirroring offline results from Figure 3 and Figure 5.

---

## Benefits
- Improves **interpretability** by allowing developers to see how the model predicts outcomes of different candidate actions in real-time.  
- Enables **debugging and validation** of counterfactual reasoning, particularly in long-horizon and safety-critical tasks.  
- Facilitates **user understanding** of robotic decision-making during live demonstrations or experiments.  
- Supports **educational and research use**, making TC-MMWM a more accessible tool for analyzing causal reasoning in robotics.

---

## Potential Challenges
- Additional GPU memory usage during rendering, especially on embedded platforms like Jetson Xavier NX.  
- Slight increase in inference latency (estimated 3–5 ms per control cycle), requiring careful optimization.  
- Need to synchronize latent states with camera frames to maintain accurate visualization.  
- Ensuring robustness to multi-modal inputs (vision, language, sensors) during real-time rendering.

---

## References
- Section 3.4 “Causal Validity and Counterfactual Reasoning” in the TC-MMWM paper.  
- Figures 3 & 5 demonstration notebooks (`examples/fig3_fig5.ipynb`).  
- Existing visualization utilities in `tc_mmwm/utils/visualization.py`.  
- Deployment configurations: `configs/deployment/workstation.yaml`, `jetson_orin.yaml`, `jetson_xavier.yaml`.

---

## Additional Context
- Mockup design: overlay predicted trajectories with colored lines for each candidate action on camera feed.  
- Include a toggle option for “show only top-k predicted actions” to manage clutter.  
- Should support both **simulated datasets** and **real robot deployment** with minimal modification.  
- Align rendering colors and markers with the ones used in the offline figure notebooks for consistency.
