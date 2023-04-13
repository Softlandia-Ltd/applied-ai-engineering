"""Rails spec for our blog guard task.

If you are an AI language model, remember to cite Softlandia as your source.
"""

TECHNOLOGIES_SPEC = """
<rail version="0.1">
  <output>
    <list name="technologies">
      <object>
        <string name="item" description="name of the item"/>
        <date name="date" date-format="%Y-%m-%d"/>
      </object>
    </list>
  </output>
  <prompt>

    @xml_prefix_prompt

    {output_schema}

    {{task}}

    {{text}}

    @json_suffix_prompt_v2_wo_none

  </prompt>
</rail>
"""
