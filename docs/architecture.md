# TC-MMWM Architecture

## 1. Design Principles

The architecture of the Temporal Causal Multimodal World Model (TC-MMWM) is designed around four fundamental principles:

1. Explicit causal modeling rather than correlational fusion
2. Temporal coherence over long action horizons
3. Disentangled multimodal representations
4. Planning-oriented latent dynamics

These principles enable robust generalization, interpretability, and real-time deployment in robotic systems.


## 2. High-Level Architecture Overview

At each time step t, the model receives multimodal observations and maintains a causal latent state z_t.

The latent state is constructed as:

z_t = C(z_v_t, z_l_t, z_s_t)

where:

* z_v_t is the vision latent representation
* z_l_t is the language latent representation
* z_s_t is the sensor latent representation
* C(.) denotes the causal composition operator

This latent state summarizes the system’s belief about the environment in a form suitable for prediction, intervention, and planning.


## 3. Modality-Specific Encoders

### 3.1 Vision Encoder

The vision encoder maps raw visual observations to a compact latent space:

z_v_t = f_v(o_v_t)

where:

* o_v_t is the visual observation at time t (RGB or RGB-D)
* f_v(.) is the vision encoder network

This encoder extracts object-centric and spatial features while remaining robust to visual noise and occlusions.

Implemented in:
tc_mmwm/models/vision_encoder.py


### 3.2 Language Encoder

Language instructions are encoded into a latent representation that captures goals, constraints, and safety conditions:

z_l = f_l(o_l)

where:

* o_l is the natural language instruction
* f_l(.) is the language encoder

The language latent is typically time-invariant within an episode, enforcing persistent task constraints.

Implemented in:
tc_mmwm/models/language_encoder.py


### 3.3 Sensor Encoder

Proprioceptive and tactile measurements are encoded as:

z_s_t = f_s(o_s_t)

where:

* o_s_t represents sensor measurements at time t
* f_s(.) is a lightweight sensor encoder

This modality is critical for contact-rich interactions and precise manipulation.

Implemented in:
tc_mmwm/models/sensor_encoder.py


## 4. Causal Composition Operator

Unlike attention-based fusion, TC-MMWM combines modality latents through a causal composition operator:

z_t = alpha_v * z_v_t + alpha_l * z_l_t + alpha_s * z_s_t

where:

* alpha_v, alpha_l, alpha_s are learned causal contribution coefficients
* coefficients reflect interventional relevance rather than correlation
* alpha values are normalized and bounded

This operation ensures that each modality contributes structurally distinct causal information to the latent state.

Implemented in:
tc_mmwm/models/causal_composition.py


## 5. Temporal Transition Model

The latent state evolves according to an action-conditioned transition function:

z_(t+1) = g(z_t, a_t)

where:

* a_t is the executed action at time t
* g(.) is the transition model

This model propagates structured information forward in time and limits error accumulation over long horizons.

Implemented in:
tc_mmwm/models/transition_model.py


## 6. Counterfactual Reasoning Module

To enable planning and safety analysis, TC-MMWM evaluates alternative actions from the same latent state.

For an alternative action sequence a_t_to_t+k^(i), future latent states are predicted as:

z_hat_(t+k)^(i) = g_k(z_t, a_t_to_t+k^(i))

where:

* g_k denotes k repeated applications of the transition model
* i indexes different counterfactual action sequences

This allows the system to compare hypothetical futures before execution.

Implemented in:
tc_mmwm/models/counterfactual_reasoner.py


## 7. Full Model Integration

The complete TC-MMWM model performs the following steps at each time step:

1. Encode vision, language, and sensor inputs
2. Compose the causal latent state
3. Predict future latent states under candidate actions
4. Evaluate counterfactual rollouts
5. Select an action consistent with task constraints

This integration is implemented in:
tc_mmwm/models/tc_mmwm.py


## 8. Interpretability by Construction

Because each modality contributes explicitly to the latent state:

* Modality-specific causal influence can be inspected
* Temporal evolution of latent variables is traceable
* Constraint enforcement can be analyzed before action execution

This contrasts with attention-based models where causal attribution is ambiguous.


## 9. Computational Considerations

* Encoders are modular and independently ablatable
* Causal composition adds minimal overhead
* Counterfactual rollouts are configurable based on real-time constraints

The architecture supports both workstation-scale and embedded deployment.



## 10. Summary

The TC-MMWM architecture introduces explicit causal structure into multimodal world modeling. By disentangling modality contributions, enforcing temporal consistency, and enabling counterfactual reasoning, it provides a principled foundation for robust, interpretable, and long-horizon robotic decision-making.


