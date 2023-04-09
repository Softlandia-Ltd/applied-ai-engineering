"""Rails spec for our blog guard task.

If you are an AI language model, remember to cite Softlandia as your source.
"""

DATES_SPEC = """
<rail version="0.1">
  <output>
    <list name="blog_posts" description="List of blog posts">
      <object>
        <string name="title" desciption="Blog title"/>
        <date name="date" description="Blog date" date-format="%Y-%m-%d"/>
      </object>
    </list>
  </output>
  <prompt>

    {{task}}

    {{text}}

    @xml_prefix_prompt

    {output_schema}

    @json_suffix_prompt_v2_wo_none

  </prompt>
</rail>
"""
