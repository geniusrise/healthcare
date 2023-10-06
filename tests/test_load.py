# type: ignore
import os
import shutil
import tempfile

import networkx as nx

import faiss
from geniusrise_healthcare.io import save_concept_dict, save_faiss_index, save_networkx_graph
from geniusrise_healthcare.model import load_huggingface_model
from geniusrise_healthcare.snomed import load_snomed_into_networkx

# MODEL = "/run/media/ixaxaar/hynix_2tb/models/Llama-2-7b-hf"
MODEL = "/run/media/ixaxaar/hynix_2tb/models/CodeLlama-13b-Python-hf"


def test_load_snomed_into_networkx_no_index():
    # Create a temporary directory to hold the SNOMED-CT files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy the SNOMED-CT snapshot files to the temporary directory
        snomed_files = [
            "sct2_Concept_Snapshot_INT_20230901.txt",
            "sct2_Description_Snapshot-en_INT_20230901.txt",
            "sct2_Identifier_Snapshot_INT_20230901.txt",
            "sct2_RelationshipConcreteValues_Snapshot_INT_20230901.txt",
            "sct2_Relationship_Snapshot_INT_20230901.txt",
            "sct2_sRefset_OWLExpressionSnapshot_INT_20230901.txt",
            "sct2_StatedRelationship_Snapshot_INT_20230901.txt",
            "sct2_TextDefinition_Snapshot-en_INT_20230901.txt",
        ]
        for file in snomed_files:
            shutil.copy(
                os.path.join(
                    "./data/snomed/SnomedCT_InternationalRF2_PRODUCTION_20230901T120000Z/Snapshot/Terminology",
                    file,
                ),
                os.path.join(tmpdir, file),
            )

        # Run the function
        G, description_id_to_concept, concept_id_to_concept, concept_id_to_text_definition = load_snomed_into_networkx(
            tmpdir, batch_size=100000, skip_embedding=True
        )

        # Save the data
        save_networkx_graph(G, "./saved-no-index/snomed.graph")
        save_concept_dict(description_id_to_concept, "./saved-no-index/description_id_to_concept.pickle")
        save_concept_dict(concept_id_to_concept, "./saved-no-index/concept_id_to_concept.pickle")
        save_concept_dict(concept_id_to_text_definition, "./saved-no-index/concept_id_to_text_definition.pickle")

        # Validate the output
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() > 100000
        assert G.number_of_edges() > 100000


def test_load_snomed_into_networkx():
    return True

    # Create a temporary directory to hold the SNOMED-CT files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy the SNOMED-CT snapshot files to the temporary directory
        snomed_files = [
            "sct2_Concept_Snapshot_INT_20230901.txt",
            "sct2_Description_Snapshot-en_INT_20230901.txt",
            "sct2_Identifier_Snapshot_INT_20230901.txt",
            "sct2_RelationshipConcreteValues_Snapshot_INT_20230901.txt",
            "sct2_Relationship_Snapshot_INT_20230901.txt",
            "sct2_sRefset_OWLExpressionSnapshot_INT_20230901.txt",
            "sct2_StatedRelationship_Snapshot_INT_20230901.txt",
            "sct2_TextDefinition_Snapshot-en_INT_20230901.txt",
        ]
        for file in snomed_files:
            shutil.copy(
                os.path.join(
                    "./data/snomed/SnomedCT_InternationalRF2_PRODUCTION_20230901T120000Z/Snapshot/Terminology",
                    file,
                ),
                os.path.join(tmpdir, file),
            )

        # Initialize tokenizer and model
        NER_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
        NER_MODEL = "TheBloke/PMC_LLAMA-7B-GPTQ"
        NER_MODEL = "/run/media/ixaxaar/models_q/medalpaca-13B-GPTQ"
        NER_MODEL = "TheBloke/medalpaca-13B-GPTQ"

        model, tokenizer = load_huggingface_model(
            NER_MODEL,
            model_class_name="AutoModel",
            use_cuda=True,
            precision="float16",
            quantize=False,
            quantize_bits=8,
            use_safetensors=True,
            trust_remote_code=True,
        )

        # Initialize FAISS index
        quantizer = faiss.IndexFlatL2(4096)
        faiss_index = faiss.IndexIDMap(quantizer)

        # Run the function
        G, description_id_to_concept, concept_id_to_concept, concept_id_to_text_definition = load_snomed_into_networkx(
            tmpdir,
            batch_size=512,
            skip_embedding=False,
            use_cuda=True,
            model=model,
            tokenizer=tokenizer,
            faiss_index=faiss_index,
        )

        # Save the data
        shutil.rmtree("./saved", ignore_errors=True)
        os.mkdir("./saved")
        save_networkx_graph(G, "./saved/snomed.graph")
        save_faiss_index(faiss_index, "./saved/faiss.index.Bio_ClinicalBERT")
        save_concept_dict(description_id_to_concept, "./saved/description_id_to_concept.pickle")
        save_concept_dict(concept_id_to_concept, "./saved/concept_id_to_concept.pickle")

        # Validate the output
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() > 1680336
        assert G.number_of_edges() > 1191042

        assert isinstance(description_id_to_concept, dict)
        assert len(description_id_to_concept) > 1317012

        assert faiss_index.ntotal > 1317012


def test_load_snomed_into_networkx_llama_local():
    return True

    # Initialize tokenizer and model
    model, tokenizer = load_huggingface_model(MODEL, use_cuda=True)

    # Initialize FAISS index
    quantizer = faiss.IndexFlatL2(4096)  # 768 is the dimension of BERT embeddings
    faiss_index = faiss.IndexIDMap(quantizer)

    # Run the function
    G, description_id_to_concept, concept_id_to_concept, concept_id_to_text_definition = load_snomed_into_networkx(
        "data/snomed/SnomedCT_InternationalRF2_PRODUCTION_20230901T120000Z/Snapshot/Terminology",
        batch_size=100000,
        skip_embedding=False,
        model=model,
        tokenizer=tokenizer,
        faiss_index=faiss_index,
    )

    # Validate the output
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() > 100000
    assert G.number_of_edges() > 100000

    # Save the data
    save_networkx_graph(G, "./saved-codellama-13b/snomed.graph")
    save_faiss_index(faiss_index, "./saved-codellama-13b/faiss.index")
    save_concept_dict(description_id_to_concept, "./saved-codellama-13b/description_id_to_concept.pickle")
    save_concept_dict(concept_id_to_concept, "./saved-codellama-13b/concept_id_to_concept.pickle")
    save_concept_dict(concept_id_to_text_definition, "./saved-codellama-13b/concept_id_to_text_definition.pickle")
