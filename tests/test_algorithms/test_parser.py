from core.parser import PseudocodeParser

def test_simple_loop():
    parser = PseudocodeParser()
    code = """
    for i = 1 to n do
        x = x + 1
    end
    """
    ast = parser.parse(code)
    info = parser.extract_structure_info(ast)
    assert len(info['loops']) == 1
    assert info['loops'][0]['type'] == 'for'