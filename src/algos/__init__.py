MODEL_CLASSES = {
    "DT": None,
    "ODT": None,
    "UDT": None,
    "DummyUDT": None,
    "MPDT": None,
    "DDT": None,
    "MDDT": None,
    "DecisionMamba": None,
    "DiscreteDecisionMamba": None,
    "MDDMamba": None,
    "DecisionXLSTM": None,
    "DiscreteDecisionXLSTM": None,
    "MDDXLSTM": None,
}

AGENT_CLASSES = {
    "DT": None,
    "ODT": None,
    "UDT": None,
    "DummyUDT": None,
    "MPDT": None,
    "DDT": None,
    "MDDT": None,
    "DecisionMamba": None,
    "DiscreteDecisionMamba": None,
    "MDDMamba": None,
    "DecisionXLSTM": None,
    "DiscreteDecisionXLSTM": None,
    "MDDXLSTM": None,
}


def get_model_class(kind):
    if kind in ["DT", "ODT", "UDT"]:
        from .models.online_decision_transformer_model import OnlineDecisionTransformerModel
        MODEL_CLASSES[kind] = OnlineDecisionTransformerModel
    elif kind in ["DDT"]:
        from .models.discrete_decision_transformer_model import DiscreteDTModel
        MODEL_CLASSES[kind] = DiscreteDTModel
    elif kind in ["MDDT"]:
        from .models.multi_domain_discrete_dt_model import MultiDomainDiscreteDTModel
        MODEL_CLASSES[kind] = MultiDomainDiscreteDTModel
    elif "mamba" in kind.lower(): 
        from .models.decision_mamba import DecisionMambaModel, DiscreteDecisionMambaModel, MultiDomainDecisionMambaModel
        MODEL_CLASSES["DecisionMamba"] = DecisionMambaModel
        MODEL_CLASSES["DiscreteDecisionMamba"] = DiscreteDecisionMambaModel
        MODEL_CLASSES["MDDMamba"] = MultiDomainDecisionMambaModel
    elif "xlstm" in kind.lower(): 
        from .models.decision_xlstm import DecisionXLSTMModel, DiscreteDecisionXLSTMModel, MultiDomainDiscreteDecisionXLSTMModel
        MODEL_CLASSES["DecisionXLSTM"] = DecisionXLSTMModel
        MODEL_CLASSES["DiscreteDecisionXLSTM"] = DiscreteDecisionXLSTMModel
        MODEL_CLASSES["MDDXLSTM"] = MultiDomainDiscreteDecisionXLSTMModel
    assert kind in MODEL_CLASSES, f"Unknown kind: {kind}"
    return MODEL_CLASSES[kind]


def get_agent_class(kind):
    assert kind in AGENT_CLASSES, f"Unknown kind: {kind}"
    # lazy imports only when needed
    if kind in ["DT", "ODT"]:
        from .decision_transformer_sb3 import DecisionTransformerSb3
        AGENT_CLASSES[kind] = DecisionTransformerSb3
    elif kind in ["UDT"]:
        from .universal_decision_transformer_sb3 import UDT
        AGENT_CLASSES[kind] = UDT
    elif kind in ["DDT", "MDDT"]:
        from .discrete_decision_transformer_sb3 import DiscreteDecisionTransformerSb3
        AGENT_CLASSES[kind] = DiscreteDecisionTransformerSb3
    elif kind == "DecisionMamba":
        from .decision_mamba import DecisionMamba
        AGENT_CLASSES[kind] = DecisionMamba
    elif kind in ["DiscreteDecisionMamba", "MDDMamba"]:
        from .decision_mamba import DiscreteDecisionMamba
        AGENT_CLASSES[kind] = DiscreteDecisionMamba
    elif kind in ["DecisionXLSTM"]:
        from .decision_xlstm import DecisionXLSTM
        AGENT_CLASSES[kind] = DecisionXLSTM    
    elif kind in ["DiscreteDecisionXLSTM", "MDDXLSTM"]:
        from .decision_xlstm import DiscreteDecisionXLSTM
        AGENT_CLASSES[kind] = DiscreteDecisionXLSTM
    return AGENT_CLASSES[kind]
