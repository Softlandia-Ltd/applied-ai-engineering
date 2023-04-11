# How to train your LLM

A complete AI engineering example using the best tools and tricks of the trade.

The first example shows how to extract structured data from blog posts.

Install the requirements into a new virtual environment
```sh
pip install -r requirements.txt
```

Run the blog scraping script
```sh
python get_blog_posts.py
```

## Sample output

Ask for blog dates:
```sh
{
    "blog_posts": [
        {
            "title": "Recap of the Second Data Science Meetup",
            "date": "2022-08-18"
        },
        {
            "title": "Building NLP Solutions: Strategies and Tools",
            "date": "2022-07-07"
        },
        {
            "title": "Explaining the Cloud: A Layman's Guide",
            "date": "2022-06-15"
        },
        {
            "title": "Real-time Data Processing with Python: Technology Evaluation",
            "date": "2022-04-22"
        },
        {
            "title": "Revolutionizing Hardware Testing with Python-based Solutions",
            "date": "2022-03-10"
        },
        {
            "title": "The Rise of Applied AI Engineers and the Shift in AI Skillsets",
            "date": "2023-04-05"
        }
    ]
}
```

Ask for technologies:
```sh
{
    "technologies": [
        {
            "technology": "Real-time data processing in Python",
            "date": "2023-02-13"
        },
        {
            "technology": "Deployment experience",
            "date": "2023-02-13"
        },
        {
            "technology": "Kubernetes cluster",
            "date": "2023-02-13"
        },
        {
            "technology": "Metaflow",
            "date": "2023-02-13"
        },
        {
            "technology": "Data science infrastructure meetup",
            "date": "2023-02-13"
        },
        {
            "technology": "Spark",
            "date": "2023-02-13"
        },
        {
            "technology": "Flink",
            "date": "2023-02-13"
        },
        {
            "technology": "Bytewax",
            "date": "2023-02-13"
        }
    ]
}
```

Results vary widely by model!
