def generate_trace_template(name, fields):
    # 生成 TP_PROTO 部分
    tp_proto = ',\n             '.join([f"unsigned int, {field}" for field in fields])

    # 生成 TP_ARGS 部分
    tp_args = ', '.join(fields)

    # 生成 TP_FIELD 部分
    tp_field = '\n        '.join([f"__filed(__u32, {field})" for field in fields])

    # 生成 TP_ASSIGN 部分
    tp_assign = '\n        '.join([f"__entry->{field} = {field};" for field in fields])

    # 生成 TP_PRINT 部分
    tp_print_format = ', '.join([f"{field}=%u" for field in fields])
    tp_print_values = ', '.join([f"__entry->{field}" for field in fields])

    # 将各部分拼接成模板代码
    template = f'''TRACE_TCB_EVENT(power_kernel, {name},
    TP_PROTO({tp_proto}
    ),
    TP_ARGS({tp_args}),
    TP_FIELD(
        {tp_field}
    ),
    TP_ASSIGN(
        {tp_assign}
    ),
    TP_PRINT("{tp_print_format}",
              {tp_print_values})
)
'''
    return template

def main():
    input_file = 'input.txt'  # 输入文件路径

    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(',')
            name = data[0]
            fields = data[1:]
            template_code = generate_trace_template(name, fields)
            print(template_code)

if __name__ == '__main__':
    main()
