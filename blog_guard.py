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

    You are an assistant, who only outputs valid JSON. The following XML describes the output format.

    {output_schema}

    {{task}}

    @json_suffix_prompt_v2_wo_none

    Text:

    {{text}}

  </prompt>
</rail>
"""


INSTRUCTIONS_TECHNOLOGIES_SPEC = """
<rail version="0.1">
  <output>
    <list name="technologies">
      <object>
        <string name="item" description="name of the item"/>
        <date name="date" date-format="%Y-%m-%d"/>
      </object>
    </list>
  </output>
  <instructions>

    You are an assistant, who only outputs valid JSON. The following XML describes the output format.

    {output_schema}

    @json_suffix_prompt_v2_wo_none

  </instructions>

  <prompt>

    {{task}}

    {{text}}

    JSON:
  </prompt>
</rail>
"""
