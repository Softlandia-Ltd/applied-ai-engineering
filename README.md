# Applied AI Engineering

A complete AI engineering example using the best tools and tricks of the trade.

This repository shows how to extract structured data from blog posts. It can easily be
adapted to chat or generative tasks as well.

Install the requirements into a new virtual environment
```sh
pip install -r requirements.txt
```

Run the blog scraping script
```sh
python extract_blog_data.py
```

## Sample output

Ask for blog titles and dates:
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
            "item": "Cloud Native Solutions",
            "date": "2023-04-05"
        },
        {
            "item": "Sensor Fusion & IoT",
            "date": "2023-04-05"
        },
        {
            "item": "Software Consulting",
            "date": "2023-04-05"
        },
        {
            "item": "Kubernetes",
            "date": "2023-02-13"
        },
        {
            "item": "Python APIs",
            "date": "2023-02-13"
        },
        {
            "item": "Eventbrite",
            "date": "2023-02-13"
        }
    ]
}
```

Results vary widely by model!

Find more details in our [blog](https://softlandia.fi/en/blog/tools-of-the-ai-engineer)!
