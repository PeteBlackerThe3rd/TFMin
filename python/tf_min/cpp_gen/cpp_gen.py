"""
    TFMin v1.0 Minimal TensorFlow to C++ exporter
    ------------------------------------------

    Copyright (C) 2019 Pete Blacker, Surrey Space Centre & Airbus Defence and
    Space Ltd.
    Pete.Blacker@Surrey.ac.uk
    https://www.surrey.ac.uk/surrey-space-centre/research-groups/on-board-data-handling

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    in the LICENCE file of this software.  If not, see
    <http://www.gnu.org/licenses/>.

    ---------------------------------------------------------------------

    cpp_gen module
    --------------
    This module provides a semantic description of C++ source which can
    then be serialised into C++ code files.

"""
import os

# define the type of indent to use when generating code
indent_str = "    "


class Comment:

    def __init__(self, text, style='//'):
        self.text = text
        self.style = style

    def format(self, indent=""):

        lines = self.text.strip().split("\n")
        # output = ""

        if self.style == '//':
            output = "\n" + indent + "// " + ("\n"+indent+"// ").join(lines)
        elif self.style == '/*':
            output = indent + "/* " + ("\n"+indent+"   ").join(lines) + " */"
        elif self.style == '/**':
            output = (indent + "/** " + ("\n"+indent+"  * ").join(lines) +
                      "\n" + indent + "  */")
        elif self.style == '//--':
            output = indent + '//' + '-'*80 + '\n'
            output += indent + "// " + ("\n"+indent+"// ").join(lines) + '\n'
            output += indent + '//' + '-'*80
        else:
            output = (indent +
                      "// Error unknown comment format style \"%s\"" %
                      self.style)

        return output + "\n"


class TypeDefinition:

    def __init__(self,
                 type,
                 template=None,
                 ref=False,
                 const=False,
                 volatile=False,
                 ptr_levels=0,
                 namespace=None):

        self.type = type
        self.template = template
        self.ref = ref
        self.const = const
        self.volatile = volatile
        self.ptr_levels = ptr_levels
        self.namespace = namespace

    def template_substitution(self, template_inst, template_def):
        """
        Method to return a copy of this type definition with any changes
        of the given template substitution made.

        TODO : Currently not recurrsive, need to improve at some point
        :param template_inst: template instance
        :param template_def: template definition
        :return: substituted type
        """

        for i, t_i in enumerate(template_inst.elements):
            if (isinstance(t_i, TypeDefinition) and
                    template_def.elements[i] == self.type):

                new_type = TypeDefinition(
                    t_i.type,
                    template=t_i.template,
                    ref=self.ref or t_i.ref,
                    const=self.const or t_i.const,
                    ptr_levels=self.ptr_levels+t_i.ptr_levels,
                    namespace=self.namespace
                )
                return new_type

        return self

    def format(self, type_only=False):

        output = self.type

        if self.template is not None:
            output += self.template.format()

        if self.namespace is not None:
            output = self.namespace + "::" + output

        if self.const:
            output = "const " + output

        if self.volatile:
            output = "volatile " + output

        if not type_only:
            output += " "

        if self.ref:
            output += "&"
        elif self.ptr_levels > 0:
            output += "*" * self.ptr_levels

        return output


class TemplateInstance:

    def __init__(self, init_elements=None):
        if init_elements is not None:
            self.elements = init_elements
        else:
            self.elements = []

    def add_element(self, new_element):
        self.elements += [new_element]

    def format(self):

        element_strings = []

        for e in self.elements:
            if isinstance(e, TypeDefinition):
                element_strings += [e.format()]
            else:
                element_strings += [e]

        return "<" + ", ".join(element_strings) + ">"


class TemplateDefinition:

    def __init__(self):
        self.elements = []
        self.element_types = []

    def add_element(self, str, type="class"):
        self.elements += [str]
        self.element_types += [type]

    def format(self):
        element_strings = []

        for i, t in enumerate(self.element_types):
            if t == "const":
                element_strings += [self.elements[i]]
            else:
                element_strings += [t + " " + self.elements[i]]

        return "template <" + ", ".join(element_strings) + ">"


class Parameter:

    def __init__(self, identifier, type, default=None):
        self.identifier = identifier
        self.type = type
        self.default = default

    def format(self, with_default=True):
        output = self.type.format() + self.identifier

        if with_default and self.default is not None:
            output += " = " + self.default

        return output


class ParameterList:

    def __init__(self):
        self.parameters = []

    def add(self, new_parameter):
        self.parameters += [new_parameter]

    def template_substitution(self, template_inst, template_def):
        """
        Method to return a copy of this parameter list with any changes
        of the given template substitution made.

        :param template_inst: template instance
        :param template_def: template definition
        :return: substituted parameter list
        """

        new_list = ParameterList()

        for p in self.parameters:
            new_parameter = Parameter(
                p.identifier,
                p.type.template_substitution(template_inst, template_def),
                default=p.default
            )
            new_list.add(new_parameter)

        return new_list

    def format(self, with_defaults=True, types_only=False):
        elements = []

        for p in self.parameters:
            if types_only:
                elements += [p.type.format(type_only=True)]
            else:
                elements += [p.format(with_default=with_defaults)]

        return ", ".join(elements)


class Statement:

    def __init__(self, text, comment=None):

        # strip leading whitespace and a trailing ';' if it exists
        self.text = text
        if self.text != "":
            self.text = text.strip()
            if self.text[-1] == ';':
                self.text = self.text[:-1]

        self.comment = comment

    def format(self, indent):
        output = ""
        if self.comment is not None:
            output = self.comment.format(indent)
        output += indent + self.text + ";\n"
        return output


class IfStatement:

    def __init__(self, condition, comment=None):
        self.condition = condition
        self.comment = comment
        self.if_code = CodeBlock()
        self.else_code = CodeBlock()

    def format(self, indent):
        output = ""
        if self.comment is not None:
            output = self.comment.format(indent)
        output += indent + "if (" + self.condition + ")\n"
        output += self.if_code.format(indent)
        if len(self.else_code.statements) > 0:
            output += indent + "else\n"
            output += else_code.format(indent)

        return output


class LoopStatement:

    def __init__(self, type, condition, comment=None):
        self.type = type
        self.condition = condition
        self.comment = comment
        self.code = CodeBlock()

    def format(self, indent):
        output = ""
        if self.comment is not None:
            output = self.comment.format(indent)

        if self.type == "for" or self.type == "while":
            output += indent + self.type + " (" + self.condition + ")\n"
            output += self.code.format(indent)
        elif self.type == "do":
            output += indent + "do"
            output += self.code.format(indent)
            output += indent + "while (" + self.condition + ")\n"

        return output


class CodeBlock(Statement):

    def __init__(self):
        super().__init__("")
        self.statements = []

    def add_statement(self, new_statement):
        self.statements += [new_statement]

    def add_block(self, new_block):
        for statement in new_block.statements:
            self.statements += [statement]

    def format(self, indent, force_braces=False):

        global indent_str

        output = ""
        for s in self.statements:
            output += s.format(indent + indent_str)

        if len(self.statements) > 1 or force_braces:
            output = indent + "{\n" + output + indent + "}\n"

        return output


class ClassElement:

    def __init__(self, identifier, comment=None):
        self.access_modifier = "public"
        self.is_static = False
        self.identifier = identifier
        self.comment = comment
        self.parent_class = None

    def format_declaration(self, indent=""):
        return ("%s// Error : call to \"%s\" base class format_declaration!" %
                (indent, self.identifier))

    def format_definition(self, indent=""):
        return ("%s// Error : call to \"%s\" base class format_definition!" %
                (indent, elf.identifier))


class ClassProperty(ClassElement):

    def __init__(self, identifier, type, comment=None,
                 static=False, initial_value=None):
        super().__init__(identifier, comment=comment)

        self.type = type
        self.static = static
        self.initial_value = initial_value

    def format_declaration(self, indent=""):

        output = self.type.format() + self.identifier + ";\n"

        if self.static:
            output = "static " + output

        output = indent + output

        if self.comment is not None:
            output = self.comment.format(indent) + output

        return output

    def format_definition(self, indent=""):
        """
        Method to format the definition of a class property, this only has
        any meaning for static properties which need to have a global
        initalisation.
        :param indent:
        :return:
        """
        output = ""
        if self.static:
            if self.comment is not None:
                output += self.comment.format(indent)
            output += (indent + self.type.format + self.identifier + " = " +
                       self.initial_value + ";\n")

        return output


class ClassMethod(ClassElement):

    def __init__(self, identifier,
                 type=None,
                 comment=None,
                 constructor=False,
                 destructor=False,
                 inline=False,
                 template=None,
                 explicit_instantiations=[]):
        super().__init__(identifier, comment=comment)

        self.type = type
        self.constructor = constructor
        self.destructor = destructor
        self.inline = inline
        self.template = template
        self.explicit_instantiations = explicit_instantiations

        self.parameter_list = ParameterList()
        self.code_block = CodeBlock()
        self.initialiser_list = []

    def format_declaration(self, indent=""):
        output = (self.identifier + "(" +
                  self.parameter_list.format(with_defaults=True) + ")")

        if not self.constructor and not self.destructor:
            output = self.type.format() + output

        if self.destructor:
            output = "~" + output

        output = indent + output

        if self.inline and len(self.initialiser_list) > 0:
            output += self.format_initialiser_list(" " * len(output))

        if self.template is not None:
            output = indent + self.template.format() + "\n" + output

        if self.comment is not None:
            output = self.comment.format(indent) + output

        if self.inline:
            output += "\n"
            output += self.code_block.format(indent, force_braces=True)
        else:
            output += ";\n"

        return output

    def format_definition(self, indent=""):

        if self.inline:
            return ""

        output = (self.identifier + "(" +
                  self.parameter_list.format(with_defaults=False) + ")")

        if self.destructor:
            output = "~" + output

        output = self.parent_class.identifier + "::" + output

        if not self.constructor and not self.destructor:
            output = self.type.format() + output

        output = indent + output

        if len(self.initialiser_list) > 0:
            output += self.format_initialiser_list(" " * len(output))

        if self.template is not None:
            output = indent + self.template.format() + "\n" + output

        if self.comment is not None:
            output = self.comment.format(indent) + output

        output += "\n" + self.code_block.format(indent, force_braces=True)

        for ei in self.explicit_instantiations:

            substituted_params = self.parameter_list.template_substitution(
              ei,
              self.template
            )

            ei_text = (indent + "template " + self.type.format() +
                       self.parent_class.identifier + "::" + self.identifier)
            ei_text += (ei.format() + "(" +
                        substituted_params.format(types_only=True) + ");\n")
            output += ei_text

        return "\n" + output

    def format_initialiser_list(self, indent=""):

        output = " : "
        indent += "   "

        output += (", \n" + indent).join(self.initialiser_list)

        return output


class ClassDef(ClassElement):

    def __init__(self, identifier, comment=None,
                 super_classes=[], template_def=None):
        super().__init__(identifier, comment=comment)

        self.super_classes = super_classes
        self.template_def = template_def

        self.elements = []

    def add(self, new_element):
        new_element.parent_class = self
        self.elements += [new_element]

    def element_by_identifier(self, identifier):

        for element in self.elements:
          if element.identifier == identifier:
            return element

    def format_declaration(self, indent=""):
        """
        Method to format the declaration of this class, defines its name and
        member functions and properties
        :param indent: the indentation level outside of this class declaration
        :return: the formatted C++ text of the declaration
        """
        global indent_str

        output = indent + "class " + self.identifier

        if self.comment is not None:
            output = self.comment.format(indent) + output

        if len(self.super_classes) > 0:
            super_class_strings = []
            for s in self.super_classes:
                super_class_strings += ["public " + s]
            output += " : " + ", " .join(super_class_strings)

        output += "\n" + indent + "{\n" + indent + "public:\n"

        public_elements = self.get_elements_by_access_modifier("public")
        for pu_e in public_elements:
            output += pu_e.format_declaration(indent+indent_str)
            output += "\n"

        protected_elements = self.get_elements_by_access_modifier("protected")
        if len(protected_elements) > 0:
            output += indent + "protected:\n"
            for pr_e in protected_elements:
                output += pr_e.format_declatation(indent+indent_str)
                output += "\n"

        private_elements = self.get_elements_by_access_modifier("private")
        if len(private_elements) > 0:
            output += indent + "private:\n"
            for pri_e in private_elements:
                output += pri_e.format_declaration(indent+indent_str)
                output += "\n"

        output += indent + "};\n"

        return "\n" + output

    def format_definition(self, indent=""):
        """
        Method to format the definition of this class, declares all non-inline
        methods, and initialises any static member variables.
        :param indent: the indentation level outside of this class declaration
        :return: the formatted C++ text of the definition
        """
        output = ""

        for e in self.elements:
            if isinstance(e, ClassMethod) and not e.inline:
                output += e.format_definition(indent)
                output += "\n"
            elif isinstance(e, ClassProperty) and e.static:
                output += e.format_definition(indent)
                output += "\n"

        return output

    def get_elements_by_access_modifier(self, access_modifier):
        selected_elements = []

        for e in self.elements:
            if e.access_modifier == access_modifier:
                selected_elements += [e]

        return selected_elements


class Dependency:

    def __init__(self, path, type='system'):
        self.path = path
        self.type = type

    def format(self):
        if self.type == 'system':
            return "#include <%s>\n" % self.path
        elif self.type == 'local':
            return "#include \"%s\"\n" % self.path


class SourcePair:

    def __init__(self, base_name):
        self.base_name = base_name
        self.dependencies = []
        self.definition_dependencies = []
        self.comment_cpp = None
        self.comment_h = None
        self.classes = []

    def write(self, path):

        header = "#ifndef __" + self.base_name.upper() + "_H__\n"
        header += "#define __" + self.base_name.upper() + "_H__\n"
        if self.comment_h is not None:
            header += self.comment_h.format()
        for d in self.dependencies:
            header += d.format()
        for c in self.classes:
            header += c.format_declaration()
        header += "\n#endif // __" + self.base_name.upper() + "_H__"

        with open(os.path.join(path, self.base_name+".h"), "w") as file:
            file.write(header)

        source = ""
        if self.comment_cpp is not None:
            source += self.comment_cpp.format()
        for d in self.definition_dependencies:
            source += d.format()
        source += Dependency(self.base_name+".h", 'local').format()
        for c in self.classes:
            source += c.format_definition()

        with open(os.path.join(path, self.base_name+".cpp"), "w") as file:
            file.write(source)
