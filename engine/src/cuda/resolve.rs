use super::*;

impl Card {
    pub(super) fn gadgets(&self, calls: &[Call], mine: &[usize]) -> Res<()> {
        let mut solves = self.solves.lock();
        for &i in mine {
            let Call::Gadget { solve, resolver, q, terminate } = &calls[i] else {
                unreachable!("gadget shard holds only gadget calls")
            };
            if *resolver > 1 || q.len() != terminate.len() || q.is_empty() || q.len() > crate::pbs::MAX_CONFIG_SUPPORT
                || q.iter().any(|x| !x.is_finite() || *x <= 0.0)
                || (q.iter().sum::<f32>() - 1.0).abs() > 2e-4
                || terminate.iter().any(|x| !x.is_finite())
            {
                return Err("invalid gadget input".into());
            }
            let slot = self.slot(&mut solves, *solve);
            slot.ngadget = q.len();
            slot.resolver = *resolver;
            let n = q.len();
            let mut data = vec![0.0f32; G_FIELDS * n];
            data[G_Q * n..(G_Q + 1) * n].copy_from_slice(q);
            data[G_TERM * n..(G_TERM + 1) * n].copy_from_slice(terminate);
            data[G_CUR_T * n..(G_CUR_T + 1) * n].fill(0.5);
            data[G_CUR_F * n..(G_CUR_F + 1) * n].fill(0.5);
            slot.gadget.put(&self.stream, 0, &data)?;
        }
        Ok(())
    }

    pub(super) fn read(&self, calls: &[Call], mine: &[usize], out: &mut Vec<(usize, Reply)>) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        let solves: Vec<usize> = mine.iter().map(|&i| calls[i].solve()).collect();
        self.lay(&solves)?;
        let fields = |call: &Call| match call {
            Call::ReadPlay { touched, focus, focus_n, cells, actual, next_cap, .. } =>
                (*touched, *focus, *focus_n, *cells, *actual, *next_cap),
            Call::ReadRefresh { touched, focus, focus_n, cells, .. }
            | Call::ReadTarget { touched, focus, focus_n, cells, .. } =>
                (*touched, *focus, *focus_n, *cells, u32::MAX, [0, 0]),
            _ => unreachable!("read shard holds only read calls"),
        };
        let touched: Vec<i32> = mine
            .iter()
            .map(|&i| {
                let t = fields(&calls[i]).0;
                (t[0] as i32) | ((t[1] as i32) << 1)
            })
            .collect();
        let focus: Vec<u32> = mine.iter().map(|&i| fields(&calls[i]).1).collect();
        let actual: Vec<u32> = mine.iter().map(|&i| fields(&calls[i]).4).collect();
        let caps: Vec<u32> = mine.iter().flat_map(|&i| fields(&calls[i]).5).collect();
        let mut offsets = Vec::with_capacity(mine.len());
        let mut total = 0usize;
        for &i in mine {
            offsets.push(total as u32);
            let (_, _, n, cells, _, cap) = fields(&calls[i]);
            total += 1 + 2 * (n[0] + n[1] + cap[0] + cap[1]) as usize + cells as usize;
        }
        let mut b = self.batch.lock();
        b.touched.put(&self.stream, touched.len(), copy(&touched))?;
        b.focus.put(&self.stream, focus.len(), copy(&focus))?;
        b.actual.put(&self.stream, actual.len(), copy(&actual))?;
        b.carry_cap.put(&self.stream, caps.len(), copy(&caps))?;
        b.out_at.put(&self.stream, offsets.len(), copy(&offsets))?;
        self.finish(&b, b.all())?;
        let mut scratch = self.scratch.lock();
        let gathered = scratch.gathered.room(total)?;
        unsafe {
            self.stream
                .launch_builder(&self.k.choose_gather)
                .arg(b.trees.buf()).arg(b.focus.buf()).arg(b.actual.buf())
                .arg(b.carry_cap.buf()).arg(b.out_at.buf()).arg(&mut *gathered)
                .launch_unit(Self::grid(b.parts, 1))
        }
        .map_err(err)?;
        let host = self.down_f.lock().recv(&self.stream, &gathered.slice(..total))?;
        for (part, &i) in mine.iter().enumerate() {
            let start = offsets[part] as usize;
            let end = offsets.get(part + 1).map_or(total, |&x| x as usize);
            out.push((i, Reply { a: host[start..end].to_vec(), ..Default::default() }));
        }
        Ok(())
    }

    pub(super) fn gadget_seed(&self, b: &Batch) -> Res<()> {
        unsafe {
            self.stream
                .launch_builder(&self.k.gadget_seed)
                .arg(b.trees.buf())
                .launch_unit(Self::grid(b.parts, 1))
        }
        .map_err(err)
    }

    pub(super) fn gadget_update(&self, b: &Batch, iter: i32, k: Cfr) -> Res<()> {
        unsafe {
            self.stream
                .launch_builder(&self.k.gadget_update)
                .arg(b.trees.buf()).arg(&iter).arg(&k.alpha).arg(&k.beta).arg(&k.gamma)
                .launch_unit(Self::grid(b.parts, 1))
        }
        .map_err(err)
    }

}
