from pathlib import Path
import re
import html
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Preformatted
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parent
pdfmetrics.registerFont(TTFont('ArialVN', 'C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialVN-Bold', 'C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFont(TTFont('ArialVN-Italic', 'C:/Windows/Fonts/ariali.ttf'))
pdfmetrics.registerFont(TTFont('ConsolasVN', 'C:/Windows/Fonts/consola.ttf'))
pdfmetrics.registerFontFamily('ArialVN', normal='ArialVN', bold='ArialVN-Bold', italic='ArialVN-Italic', boldItalic='ArialVN-Bold')
INK=HexColor('#17313E'); TEAL=HexColor('#087E8B'); GRAY=HexColor('#536770'); PALE=HexColor('#EDF5F7')
W,H=A4; CW=W-96
styles={
 'h1':ParagraphStyle('h1',fontName='ArialVN-Bold',fontSize=18,leading=23,textColor=INK,spaceAfter=15,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='ArialVN-Bold',fontSize=14,leading=19,textColor=TEAL,spaceAfter=12,keepWithNext=True),
 'h3':ParagraphStyle('h3',fontName='ArialVN-Bold',fontSize=10.5,leading=14,textColor=TEAL,spaceBefore=8,spaceAfter=5,keepWithNext=True),
 'body':ParagraphStyle('body',fontName='ArialVN',fontSize=9.3,leading=13.4,textColor=INK,spaceAfter=6,splitLongWords=True),
 'bullet':ParagraphStyle('bullet',fontName='ArialVN',fontSize=9.3,leading=13.4,textColor=INK,spaceAfter=5,leftIndent=11,firstLineIndent=-9),
 'table':ParagraphStyle('table',fontName='ArialVN',fontSize=8.45,leading=11.4,textColor=INK,spaceAfter=0,splitLongWords=True),
 'th':ParagraphStyle('th',fontName='ArialVN-Bold',fontSize=8.45,leading=11.4,textColor=white,spaceAfter=0),
 'code':ParagraphStyle('code',fontName='ConsolasVN',fontSize=7.4,leading=10.9,textColor=INK,spaceAfter=7,leftIndent=8,backColor=PALE,borderPadding=8),
}

def markup(s):
    s=html.escape(s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'`([^`]+)`',r'<font name="ConsolasVN" size="8.2">\1</font>',s)
    s=re.sub(r'(https://[^\s]+)',lambda m:'<link href="'+m.group(1)+'" color="#087E8B">'+m.group(1)+'</link>',s)
    return s

def para(s,style='body'):
    return Paragraph(markup(s),styles[style])

def parse(md):
    lines=md.splitlines(); out=[]; i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line: i+=1; continue
        if line=='<!-- PAGEBREAK -->': out.append(PageBreak()); i+=1; continue
        if line.startswith('```'):
            code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'): code.append(lines[i]);i+=1
            out.append(Preformatted('\n'.join(code),styles['code']));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',x) for x in cells): rows.append(cells)
                i+=1
            n=len(rows[0]); widths={2:[CW*.31,CW*.69],3:[CW*.22,CW*.35,CW*.43]}[n]
            if rows[0][0]=='Mốc tương đối': widths=[CW*.22,CW*.15,CW*.63]
            if rows[0][0]=='Tệp': widths=[CW*.25,CW*.75]
            data=[[para(c,'th' if j==0 else 'table') for c in r] for j,r in enumerate(rows)]
            table=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),TEAL),('ROWBACKGROUNDS',(0,1),(-1,-1),[PALE,white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,-1),(-1,-1),.5,HexColor('#B3CDD3'))]))
            out.extend([table,Spacer(1,6)]);continue
        if line.startswith('### '): out.append(para(line[4:],'h3'))
        elif line.startswith('## '): out.append(para(line[3:],'h2'))
        elif line.startswith('# '): out.append(para(line[2:],'h1'))
        elif line.startswith('- '): out.append(para('• '+line[2:],'bullet'))
        elif re.match(r'^\d+\. ',line): out.append(para(line,'bullet'))
        else:
            block=[line]
            while i+1<len(lines) and lines[i+1].strip() and not lines[i+1].startswith(('#','|','- ','```','<!--')):
                i+=1;block.append(lines[i].strip())
            out.append(para(' '.join(block)))
        i+=1
    return out

class NumberedCanvas(canvas.Canvas):
    def __init__(self,*a,**kw):
        canvas.Canvas.__init__(self,*a,**kw);self.states=[]
    def showPage(self):
        self.states.append(dict(self.__dict__));self._startPage()
    def save(self):
        count=len(self.states)
        for state in self.states:
            self.__dict__.update(state); self.setFillColor(INK);self.setFont('ArialVN',7.5)
            self.drawString(48,28,'CV NÂNG CAO  |  BẢN TRIỂN KHAI NHÓM  |  21/10/2026')
            self.drawRightString(W-48,28,f'{self._pageNumber:02d} / {count:02d}')
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

def page_header(c,doc):
    c.saveState();c.setFillColor(TEAL);c.rect(0,H-7,W,7,fill=1,stroke=0)
    c.setFont('ArialVN-Bold',7.2);c.drawString(48,H-30,'NHẬN DIỆN KHUÔN MẶT KHI BỊ CHE KHUẤT')
    c.setFont('ArialVN',7.2);c.setFillColor(GRAY);c.drawRightString(W-48,H-30,'A / B / C')
    c.setStrokeColor(HexColor('#CEE0E5'));c.line(48,43,W-48,43);c.restoreState()

if __name__=='__main__':
    src=ROOT/'huong_dan_trien_khai_hoan_chinh.md'
    target=ROOT/'huong_dan_trien_khai_hoan_chinh.pdf'
    doc=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=49,bottomMargin=53,title='Nhận diện khuôn mặt khi bị che khuất - Hướng dẫn nhóm',author='Nhóm CV nâng cao',pageCompression=1)
    doc.build(parse(src.read_text(encoding='utf-8')),onFirstPage=page_header,onLaterPages=page_header,canvasmaker=NumberedCanvas)
    print('Created PDF, bytes:',target.stat().st_size)
