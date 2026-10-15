"""Small executable regression set for the language pipeline.

These tests intentionally exercise the parser, generator based execution engine,
ProgramState memory, and generic control-flow events together.
"""
import unittest

from app import Interpreter, Parser, tokenize


def run(source, inputs=None):
    engine = Interpreter(Parser(tokenize(source)).parse_program(), inputs or [])
    events = []
    while True:
        step = engine.step()
        if step is None:
            break
        events.extend(step)
    returns = [e["value"] for e in events if e["type"] == "FUNCTION_RETURN" and e["name"] == "main"]
    return engine.state, events, (returns[-1] if returns else None)


class LanguageRegressionTests(unittest.TestCase):
    def test_arithmetic_if_and_scope(self):
        state, _, result = run("int main(){int x=2; {int y=3; x=x+y;} if(x==5){return x;} return 0;}")
        self.assertEqual(result, 5)
        self.assertTrue(state.done)

    def test_for_continue_and_break(self):
        _, events, result = run("int main(){int s=0; for(int i=0;i<5;i++){if(i==1)continue; if(i==4)break; s=s+i;} return s;}")
        self.assertEqual(result, 5)
        self.assertIn("CONTINUE", [e["type"] for e in events])
        self.assertIn("BREAK", [e["type"] for e in events])

    def test_while_and_do_while(self):
        _, _, result = run("int main(){int i=0; do{i++;}while(i<2); while(i<4)i++; return i;}")
        self.assertEqual(result, 4)

    def test_switch_default_and_fallthrough(self):
        _, events, result = run("int main(){int x=1;int y=0;switch(x){case 1:y=y+1;case 2:y=y+10;break;default:y=99;}return y;}")
        self.assertEqual(result, 11)
        kinds = [e["type"] for e in events]
        for kind in ("SWITCH_ENTER", "CASE_CHECK", "CASE_MATCH", "CASE_EXECUTION", "SWITCH_EXIT"):
            self.assertIn(kind, kinds)

    def test_runtime_sized_arrays(self):
        _, _, result = run("int main(){int n=5;int arr[n];arr[0]=10;arr[1]=20;arr[2]=30;arr[3]=40;arr[4]=50;return arr[4];}")
        self.assertEqual(result, 50)
        _, _, result = run("int main(){int a=2;int b=3;int arr[a+b];arr[4]=99;return arr[4];}")
        self.assertEqual(result, 99)

    def test_arrays_pointers_and_functions(self):
        source = "int add(int a,int b){return a+b;} int main(){int a[2]={3,4};int *p=&a[0];return add(*p,a[1]);}"
        _, events, result = run(source)
        self.assertEqual(result, 7)
        self.assertIn("FUNCTION_CALL", [e["type"] for e in events])
        self.assertIn("POINTER_DEREFERENCE", [e["type"] for e in events])

    def test_string_input_and_printf(self):
        state, events, result = run('int main(){char str[100];scanf("%s",str);printf("%s",str);return 0;}', ["hello"])
        self.assertEqual(state.output, "hello")
        self.assertEqual(result, 0)
        self.assertIn("INPUT_RECEIVED", [e["type"] for e in events])

    def test_strlen_and_strcmp(self):
        _, _, length = run('int main(){char a[100],b[100];scanf("%s",a);scanf("%s",b);return strlen(a);}', ["hello", "world"])
        self.assertEqual(length, 5)
        _, _, equal = run('int main(){char a[100],b[100];scanf("%s",a);scanf("%s",b);return strcmp(a,b);}', ["hello", "hello"])
        self.assertEqual(equal, 0)

    def test_char_index_and_literal(self):
        _, _, result = run("int main(){char s[100];scanf(\"%s\",s);if(s[0]=='a')return 1;return 0;}", ["apple"])
        self.assertEqual(result, 1)

    def test_generic_palindrome_flow(self):
        source = ('int main(){char str[100],rev[100];int i,len;scanf("%s",str);'
                  'len=strlen(str);for(i=0;i<len;i++){rev[i]=str[len-i-1];}'
                  'rev[len]=\'\\0\';if(strcmp(str,rev)==0){printf("Palindrome");return 1;}'
                  'printf("Not a palindrome");return 0;}')
        state, _, palindrome = run(source, ["madam"])
        self.assertEqual(state.output, "Palindrome")
        self.assertEqual(palindrome, 1)


if __name__ == "__main__":
    unittest.main()
