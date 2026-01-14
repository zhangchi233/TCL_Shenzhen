class HintFactory:
    """
    根据 L3 Tag 和 Image Type 生成针对性的 Question 生成指引。
    """
    
    @staticmethod
    def get_domain_hint(tag: str) -> str:
        """
        根据 tag 返回对应的中文领域提示（domain hint）
        """
        if not tag:
            return ""

        tag_lower = tag.lower()

        # --- 显示技术（Display） ---
        if "micro-led" in tag_lower or "microled" in tag_lower:
            domain_hint = (
                "具体询问像素结构、钝化层，或（如果能看出来）所采用的巨量转移方法；"
                "留意是否提到侧壁缺陷（sidewall defects）或外量子效率（EQE）。"
            )

        elif "oled" in tag_lower:
            domain_hint = (
                "询问发光层（EML）、载流子注入层（HIL / EIL），"
                "或具体的有机材料堆叠结构。"
            )

        elif "lcd" in tag_lower or "liquid crystal" in tag_lower:
            domain_hint = (
                "重点关注液晶取向方式、背光模组（BLU），"
                "或偏光片的排列结构。"
            )

        # --- 背板（TFT） ---
        elif "tft" in tag_lower or "transistor" in tag_lower:
            domain_hint = (
                "重点关注沟道材料（IGZO / LTPS / a-Si）、"
                "栅极结构（顶栅 / 底栅），以及源极/漏极接触区域。"
            )

        # --- 工艺（Process） ---
        elif "deposition" in tag_lower or "cvd" in tag_lower or "pvd" in tag_lower:
            domain_hint = (
                "询问薄膜的均匀性、所使用的前驱体气体，"
                "或工艺过程中的温度曲线。"
            )

        elif "lithography" in tag_lower or "etching" in tag_lower:
            domain_hint = (
                "重点关注关键尺寸（CD）、光刻胶类型，"
                "或刻蚀工艺的选择比（selectivity）。"
            )

        elif "encap" in tag_lower or "tfe" in tag_lower:
            domain_hint = (
                "询问阻隔层结构（无机 / 有机交替层），"
                "以及水汽透过率（WVTR）带来的影响。"
            )

        # --- 评价与缺陷（Characterization & Defects） ---
        elif "gamut" in tag_lower or "color" in tag_lower:
            domain_hint = (
                "询问 CIE 色度坐标、NTSC / Rec.2020 色域覆盖率，"
                "或是否存在色偏（color shift）。"
            )

        else:
            domain_hint = ""

        return domain_hint
