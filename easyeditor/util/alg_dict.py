from ..models.a3e import A3EHyperParams, apply_a3e_to_model, apply_a3e_to_multimodal_model

ALG_DICT = {
    'A3E': apply_a3e_to_model,
}

ALG_MULTIMODAL_DICT = {
    'A3E': apply_a3e_to_multimodal_model
}
