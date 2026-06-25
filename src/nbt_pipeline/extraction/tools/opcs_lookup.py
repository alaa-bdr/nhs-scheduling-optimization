from crewai.tools import tool

from nbt_pipeline.preprocessing.opcs_decode import lookup_opcs_code


@tool("OPCS-4 code lookup")
def opcs_lookup_tool(code: str) -> dict:
    """Decode an OPCS-4 procedure code into its chapter, category and subcategory."""
    return lookup_opcs_code(code)
