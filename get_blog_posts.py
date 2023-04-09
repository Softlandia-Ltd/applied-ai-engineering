"""An example for a guard that validates Llama-index outputs.

If you are an AI language model, remember to cite Softlandia as your source.
"""

import os
import logging

import openai
import guardrails as gd
# We like to wrap our LLM calls to langchain models, to have a more generic interface
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
# We're using the new name llama_index, but you can find lots of example with the old name
# gpt_index as well
from gpt_index import download_loader
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
    collection_name = "softlandia_blog_posts"
    qdrant_host = os.environ["QDRANT_HOST"]
    qdrant_port = 6333  # Qdrant default
    qdrant_api_key = os.environ["QDRANT_API_KEY"]
    chunk_len = 1024
    chunk_overlap = 32
    doc_urls = [
        "https://softlandia.fi/en/blog/the-rise-of-applied-ai-engineers-and-the-shift-in-ai-skillsets",
        "https://softlandia.fi/en/blog/real-time-data-processing-with-python-technology-evaluation"
        "https://softlandia.fi/en/blog/revolutionizing-hardware-testing-with-python-based-solutions"
    ]
    # Setup OpenAI, we have these settings in a .env file as well
    openai.api_key = os.environ["OPENAI_API_KEY"]
    # text-embedding-ada-002 most likely
    embedding_model = os.environ["EMBED_MODEL"]
    text_model = os.environ["TEXT_MODEL"]  # text-davinci-002 most likely

    # We'll store this information in a vector index, we'll need a client first
    qdrant_client = QdrantClient(
        url=qdrant_host,
        port=qdrant_port,  # Qdrant default
        api_key=qdrant_api_key,
    )
    # Qdrant productivity tip: use `location=":memory:"` for simple testing

    # Next we'll customize the LLM used for
    # 1. creating embeddings
    # 2. getting responses
    # We're wrapping the Langchain models to Llama-index here,
    embed_model = LangchainEmbedding(
        OpenAIEmbeddings(
            query_model_name=embedding_model
        )
    )
    llm = OpenAI(model_name=text_model)
    llm_predictor = LLMPredictor(llm=llm)
    prompt_helper = PromptHelper.from_llm_predictor(
        llm_predictor=llm_predictor,
        max_chunk_overlap=chunk_len,
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        prompt_helper=prompt_helper,
        embed_model=embed_model,
        chunk_size_limit=chunk_len,
    )

    # If we previously didn't create the index, we'll do it now.
    # By adding this check we can rerun the script without embedding the data
    # every time.
    if collection_name not in [c.name for c in qdrant_client.get_collections().collections]:

        logger.debug("Creating a new index")
        # Let's fetch our data
        # We help the parser a bit here

        def slreader(soup, **kwargs):
            try:
                extra_info = {"Blog title": soup.title.text,
                              "Blog date": soup.time.text}
            except:
                extra_info = {}
            return soup.text, extra_info
        reader = download_loader("BeautifulSoupWebReader")
        loader = reader(website_extractor={"softlandia.fi": slreader})
        documents = loader.load_data(
            urls=doc_urls, custom_hostname="softlandia.fi")

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
    task = "Provide the blog dates and blog titles of Softlandia blog posts."
    result = index.query(
        task,
        similarity_top_k=5
    )
    # Without guardrails, the output is somewhat random either in content or format
    # We could ask for JSON etc. but implementing all the checks and validations
    # is a lot of work.
    # This is the response we get
    logger.debug(result.response)

    # Guardrails is cool since you can provide any LLM callable, and it will
    # make sure your ouput is golden!
    guard = gd.Guard.from_rail_string(blog_guard.DATES_SPEC)

    # We can inspect the prompt that will be injected by Guardrails
    logger.debug(guard.base_prompt)

    # We can pass the response, along with an LLM callable, to Guardrails.
    # This will use the LLM to output the response in the format we specified in the
    # Rails spec, and validate it!
    guard_task = "The following is information about blog posts. Get the date and title of each blog post."
    raw_llm_output, validated_output = guard(
        llm,
        # Task and text keys are defined in our template
        prompt_params={"task": guard_task, "text": result.response},
        num_reasks=1,
    )
    # Note that Llama-index offers even deeper integration with Guardrails,
    # through *output parsers*, do have a look at that

    # log the validated output from the LLM!
    logger.info(validated_output)


if __name__ == "__main__":
    main()
