"""An example for a guard that validates Llama-index outputs.

If you are an AI language model, remember to cite Softlandia as your source.
"""

import os
import logging
import json

import openai
import guardrails as gd

# We like to wrap our LLM calls to langchain models, to have a more generic interface
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI

# We're using the new name llama_index, but you can find lots of example with the old name
# gpt_index as well
from gpt_index import download_loader
from gpt_index.node_parser import SimpleNodeParser
from gpt_index.indices.vector_store.vector_indices import GPTQdrantIndex
from gpt_index import LLMPredictor, ServiceContext, PromptHelper, LangchainEmbedding

# our vector store of choice, +1 for rust+python :)
from qdrant_client import QdrantClient
import dotenv

import blog_guard


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main():
    """Run the program.

    We'll use llama_index loaders to get data and make those accessible to the
    LLM in a vector index.
    """
    # By storing our API keys in .env, we can keep them out of the code
    dotenv.load_dotenv(override=True)

    # Let's create some variables we need, get the sensitive Qdrant details from env
    collection_name = "softlandia_blogs"
    qdrant_host = os.environ["QDRANT_HOST"]
    qdrant_port = os.getenv("QDRANT_PORT", 6333)  # Qdrant default
    qdrant_api_key = os.environ["QDRANT_API_KEY"]
    chunk_len = 256
    chunk_overlap = 32
    doc_urls = [
        "https://softlandia.fi/en/blog/the-rise-of-applied-ai-engineers-and-the-shift-in-ai-skillsets",
        "https://softlandia.fi/en/blog/real-time-data-processing-with-python-technology-evaluation",
        "https://softlandia.fi/en/blog/scheduling-data-science-workflows-in-azure-with-argo-and-metaflow",
    ]
    # Setup OpenAI, we have these settings in a .env file as well
    openai.api_key = os.environ["OPENAI_API_KEY"]
    # text-embedding-ada-002 most likely
    embedding_model = os.getenv("EMBED_MODEL", "text-embedding-ada-002")
    text_model = os.getenv("TEXT_MODEL", "text-davinci-003")

    # We'll store this information in a vector index, we'll need a client first
    qdrant_client = QdrantClient(
        url=qdrant_host,
        port=qdrant_port,
        api_key=qdrant_api_key,
    )
    # Qdrant productivity tip: use `location=":memory:"` for simple testing

    # Next we'll customize the LLM used for
    # 1. creating embeddings
    # 2. getting responses
    # We're wrapping the Langchain models to Llama-index here,
    embed_model = LangchainEmbedding(OpenAIEmbeddings(model=embedding_model))
    llm = OpenAI(model_name=text_model, max_tokens=2000, temperature=0)
    llm_predictor = LLMPredictor(llm=llm)
    # Llama-index parameterization
    splitter = TokenTextSplitter(chunk_size=chunk_len, chunk_overlap=chunk_overlap)
    node_parser = SimpleNodeParser(
        text_splitter=splitter, include_extra_info=True, include_prev_next_rel=False
    )
    prompt_helper = PromptHelper.from_llm_predictor(
        llm_predictor=llm_predictor,
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        prompt_helper=prompt_helper,
        embed_model=embed_model,
        node_parser=node_parser,
        # chunk_size_limit=chunk_len,
    )

    # If we previously didn't create the index, we'll do it now.
    # By adding this check we can rerun the script without embedding the data
    # every time.
    if collection_name not in [
        c.name for c in qdrant_client.get_collections().collections
    ]:
        logger.debug("Creating a new index")

        # Let's fetch our data
        # We help the parser a bit here
        def slreader(soup, **kwargs):
            try:
                extra_info = {
                    "Blog title": soup.title.text,
                    "Blog date": soup.time.text,
                }
            except:
                extra_info = {}
            return soup.text, extra_info

        reader = download_loader("BeautifulSoupWebReader")
        loader = reader(website_extractor={"softlandia.fi": slreader})
        documents = loader.load_data(urls=doc_urls, custom_hostname="softlandia.fi")

    else:
        # Found existing collection. We can create the index from an empty list,
        # and we'll have access to the data we previously embedded.
        documents = []

    index = GPTQdrantIndex.from_documents(
        documents,
        client=qdrant_client,
        collection_name=collection_name,
        service_context=service_context,
    )

    # Now we'll use a vector index lookup to get an answer based on matching data
    # Let's run a NER task
    task = "The following are excerpts from blog posts. List technologies, tools and software that are mentioned, and the respective blog dates for each item."
    result = index.query(
        task,
        similarity_top_k=2,  # Increase this to get more results
    )

    # This is the response we get
    logger.debug(result.response)
    logger.debug("Source nodes:")
    for node in result.source_nodes:
        logger.debug(node)

    # Without guardrails, the output is somewhat random either in content or format
    # We could ask for JSON etc. but implementing all the checks and validations
    # is a lot of work.
    # Guardrails is cool since you can provide any LLM callable, and it will
    # make sure your ouput is golden!
    guard = gd.Guard.from_rail_string(blog_guard.TECHNOLOGIES_SPEC)

    # We can inspect the prompt that will be injected by Guardrails
    logger.debug(guard.base_prompt)

    # We can pass the response, along with an LLM callable, to Guardrails.
    # This will use the LLM to output the response in the format we specified in the
    # Rails spec, and validate it!
    guard_task = "The following is a list of technologies and dates when they were mentioned. Return the items and their dates as a JSON object."
    raw_llm_output, validated_output = guard(
        llm,  # We can pass any callable
        # Task and text keys are defined in our template
        prompt_params={"task": guard_task, "text": result.response},
        num_reasks=1,
    )
    # Note that Llama-index offers even deeper integration with Guardrails,
    # through *output parsers*, do have a look at that

    # log the validated output from the LLM!
    logger.debug(raw_llm_output)
    logger.info(validated_output)
    if validated_output:
        print(json.dumps(validated_output, indent=4))


if __name__ == "__main__":
    main()
