TRACE_TCB_EVENT(power_kernel, name1,
    TP_PROTO(unsigned int, big_cpu_freq,
             unsigned int, little_cpu_freq,
             unsigned int, big_power,
             unsigned int, little_power
    ),
    TP_ARGS(big_cpu_freq, little_cpu_freq, big_power, little_power),
    TP_FIELD(
        __filed(__u32, big_cpu_freq)
        __filed(__u32, little_cpu_freq)
        __filed(__u32, big_power)
        __filed(__u32, little_power)
    ),
    TP_ASSIGN(
        __entry->big_cpu_freq = big_cpu_freq;
        __entry->little_cpu_freq = little_cpu_freq;
        __entry->big_power = big_power;
        __entry->little_power = little_power;
    ),
    TP_PRINT("big_cpu_freq=%u, little_cpu_freq=%u, big_power=%u, little_power=%u",
              __entry->big_cpu_freq, __entry->little_cpu_freq, __entry->big_power, __entry->little_power)
)