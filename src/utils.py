import platform
import sys
import select

if platform.system() == 'Windows':
    import msvcrt

class TerminalInputReader():
    def __init__(self):
        if platform.system() == 'Windows':
            self.windows = True
            self.buffer = []
        else:
            self.windows = False
            self.poller = select.poll()
            self.poller.register(sys.stdin, select.POLLIN)


    def poll(self) -> str | None:
        if self.windows:
            return self.poll_windows()
        else:
            return self.poll_unix()
        
    def poll_windows(self):
        if msvcrt.kbhit():
            ch = msvcrt.getche()
            if ch == b'\r':
                ret = ''.join(self.buffer)
                self.buffer = []
                print('')
                return ret.strip()
            elif ch == b'\x08':
                self.buffer = self.buffer[:-1]
                sys.stdout.write(' \b')
                sys.stdout.flush()
            else:
                try:
                    self.buffer.append(ch.decode('utf-8'))
                except UnicodeDecodeError:
                    return

    def poll_unix(self):
        events = self.poller.poll(0)

        if events:
            for fd, event in events:
                if fd == sys.stdin.fileno() and event & select.POLLIN:
                    line = sys.stdin.readline().strip()
                    if line:
                        return line.strip()